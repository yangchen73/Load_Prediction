import pandas as pd
import numpy as np
import argparse
import torch
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import cycle
from model.model import LSTM_kan

def read_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    df = df.dropna()
    df = df.reset_index(drop = True)
    return df


def split_data(df, ratio = 0.8):
    X = df[['Power', 'max_power', 'power_t-1', 'min_power','RMS','10%_quantile_power','power_t-2','min_power','Peak_to_Peak','power_t-12','power_t-4','power_t-10', 'power_t-8', 'power_t-7' ,'approx_max' ,'power_t-3' ,'90%_quantile_power' ,'Temperature', 'Duration_Above90%', 'power_t-11', 'Duration_Below10%', 'power_t-5', 'detail_kurtosis', 'median_power', 'power_t-6', 'std_power', 'Humidity', 'approx_sd', 'FFT_Magnitude_mean', 'approx_energy', 'Month', 'Hour', 'is_holiday']]
    y = df['Power']
    
    train_size = int(len(df) * ratio)
    train_data = X.iloc[:train_size]
    train_target = y.iloc[:train_size]
    train_target =  np.array(train_target)
    train_result = df[:train_size][['Time','Power']]

    test_data = X.iloc[train_size:]
    test_target = y.iloc[train_size:]
    test_target = np.array(test_target)
    test_result = df[train_size:][['Time','Power']]
    return np.array(train_data), np.array(test_data), train_result, test_result, np.array(train_target), np.array(test_target)

# 准备训练测试数据
def prepare_data(data, target, n_steps):
    X_data, y_data = [], []
    for i in range(n_steps-1, len(data)-168):
        x = data[i - n_steps+1:i+1]
        y = target[i+1:i+169].reshape(-1, 1)     
        X_data.append(x)
        y_data.append(y)
    X_data, y_data = np.array(X_data), np.array(y_data)
    return X_data, y_data

def train_LSTM_KAN_model(model, X_train, y_train, optimiser, criterion, epochs):
    start_time = time.time()
    y_train_pred = y_train.copy()
    model.train()

    for t in range(1, epochs+1):
        x_input = torch.tensor(X_train, dtype=torch.float32)
        y_input = torch.tensor(y_train, dtype=torch.float32)
        y_pred = model(x_input)
        y_train_pred = y_pred
        loss = criterion(y_pred, y_input)  
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(f"Epoch {t}/{epochs}")
    training_time = time.time() - start_time
    print(f"Training time: {training_time}")
    return y_train_pred

def test_LSTM_KAN_model(model, X_test, y_test):
    y_test_pred = y_test.copy()
    model.eval()  
    with torch.no_grad(): 
            x_input = torch.tensor(X_test, dtype=torch.float32)
            y_pred = model(x_input)
            y_test_pred = y_pred
    return y_test_pred

def generate_predictions(results_df, predictions, step=168, front_nan=24):
    pred_list = []
    for i in range(0, len(predictions) - front_nan - 1, step):
        y_pred = predictions[i].detach().numpy().flatten()
        pred_list.append(y_pred)
    # 创建 NaN 数组填充开头和结尾
    nan_array_front = np.full(front_nan, np.nan)
    nan_array_back = np.full(len(results_df) - len(pred_list) * step - front_nan, np.nan)
    pred_array = np.hstack(pred_list)
    pred_array = np.concatenate((nan_array_front, pred_array, nan_array_back))

    results_df['predict'] = pred_array
    return results_df

def calculate_errors(actual, predicted):
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    mae = mean_absolute_error(actual_filtered, predicted_filtered)
    rmse = np.sqrt(mean_squared_error(actual_filtered, predicted_filtered))
    r2 = r2_score(actual_filtered, predicted_filtered)
    return mae, rmse, r2

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.device = torch.device('cpu')
    parser.add_argument('--path_data', type=str, default='data/office建模特征_1.csv')
    parser.add_argument('--path_result', type=str, default='result/resident_persistent_score.csv')

    parser.add_argument('--vision', type=bool, default=True)
    parser.add_argument('--input_features', type=list, default=['Power', 'max_power', 'power_t-1', 'min_power','RMS','10%_quantile_power','power_t-2','min_power','Peak_to_Peak','power_t-12','power_t-4','power_t-10', 'power_t-8', 'power_t-7' ,'approx_max' ,'power_t-3' ,'90%_quantile_power' ,'Temperature', 'Duration_Above90%', 'power_t-11', 'Duration_Below10%', 'power_t-5', 'detail_kurtosis', 'median_power', 'power_t-6', 'std_power', 'Humidity', 'approx_sd', 'FFT_Magnitude_mean', 'approx_energy', 'Month', 'Hour', 'is_holiday'])
    parser.add_argument('--output_features', type=list, default=['Power'])
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--train_test_ratio', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=34)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--grid_size', type=int, default=200,help='grid')

    parser.add_argument('--num_channels', type=list, default=[25, 50, 25])
    parser.add_argument('--kernel_size', type=int, default=3)

    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_space', type=int, default=32)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4, help='Adam learning rate')
    arg = parser.parse_args([])
    df = read_data(arg.path_data)
    df = preprocess_data(df)
    train_data, test_data, train_result, test_result, train_target, test_target = split_data(df)
    model = LSTM_kan(
        input_dim=len(arg.input_features),
        hidden_dim=arg.hidden_dim,
        num_layers=arg.n_layers,
        output_dim=len(arg.output_features)
    )
    model.to(parser.device)  
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=arg.lr)
    n_steps = 100
    X_train, y_train = prepare_data(train_data, train_target, n_steps)

    X_test, y_test = prepare_data(test_data, test_target, n_steps)
    # 训练模型
    y_train_pred = train_LSTM_KAN_model(model, X_train, y_train, optimiser, criterion, arg.num_epochs)
    # 测试模型
    y_test_pred = test_LSTM_KAN_model(model, X_test, y_test)  
   
    train_result = generate_predictions(train_result, y_train_pred)
    test_result = generate_predictions(test_result, y_test_pred)
    # 计算误差
    train_mae, train_rmse, train_r2 = calculate_errors(train_result['Power'], train_result['predict'])
    test_mae, test_rmse, test_r2 = calculate_errors(test_result['Power'], test_result['predict'])
    # 保存分数
    score = pd.DataFrame({
        'Dataset': ['Train', 'Test'],
        'MAE': [train_mae, test_mae],
        'RMSE': [train_rmse, test_rmse],
        'r2': [train_r2, test_r2]
    })
    score.to_csv((arg.path_result), index=True)


 



