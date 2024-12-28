import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# Import models
from model.LSTMKAN import LSTM_kan
from model.GRU import GRU
from model.LSTM import LSTM
from model.KAN import KAN
from model.MLP import MLP

# Configuration
loaction_type = 'resident'
model_type = 'LSTM_kan'

input_features_dict = {
    'office': ['Power', 'RMS', 'max_power', '10%_quantile_power', 'min_power', 'is_holiday', 'Hour', 'power_t-1', 
               'Duration_Above90%', '90%_quantile_power', 'power_t-2', 'approx_max', 'median_power', 
               'Peak_to_Peak', 'Duration_Below10%', 'power_t-9'],
    'commercial': ['Power', 'RMS', 'max_power', '10%_quantile_power', '90%_quantile_power', 'min_power', 'power_t-1', 
                   'median_power', 'Hour', 'power_t-12', 'power_t-9', 'Duration_Above90%', 'approx_energy', 
                   'power_t-6', 'power_t-2', 'approx_max'],
    'public': ['Power', 'power_t-1', 'RMS', 'max_power', 'min_power', '90%_quantile_power', '10%_quantile_power', 'Hour',
               'power_t-2', 'Duration_Below10%', 'power_t-12', 'power_t-5', 'Temperature', 'median_power', 'approx_max',
               'power_t-3'],
    'resident': ['Power', 'RMS', 'max_power', 'power_t-1', '90%_quantile_power', 'min_power', '10%_quantile_power', 
                 'Hour', 'median_power', 'approx_energy', 'approx_mean', 'is_holiday', 'approx_max', 'power_t-6', 
                 'FFT_Total_energy', 'power_t-7']
}

output_features = ['Power']
train_ratio = 0.8
window_size = 24
input_dim = len(input_features_dict[loaction_type]) - 1
output_dim = len(output_features)
length_size = 360
batch_size = 128
epochs = 100

# Function to set seeds for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def data_loader(window, length_size, batch_size, data_x, data_y):
    seq_len = window  
    sequence_length = seq_len + length_size  
    result_x = [] 
    result_y = [] 
    for index in range(len(data_x) - sequence_length):  
        result_x.append(data_x[index: index + sequence_length]) 
        result_y.append(data_y[index: index + sequence_length])
    result_x = np.array(result_x)  
    result_y = np.array(result_y)
    x_data = result_x[:, :-length_size] 
    y_data = result_y[:, -length_size:, -1]  
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], input_dim))  
    y_data = np.reshape(y_data, (y_data.shape[0], -1))

    x_data, y_data = torch.tensor(x_data).to(torch.float32), torch.tensor(y_data).to(
        torch.float32) 
    ds = torch.utils.data.TensorDataset(x_data, y_data)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,shuffle=True) 
    return dataloader, x_data, y_data

def model_train(model_type, length_size, patience=10):
    start_time = time.time()
    best_val_loss = float('inf') 
    epochs_no_improve = 0  
    best_model = None 

    if model_type == "GRU":
        net = GRU(input_size=input_dim, hidden_size=64, num_layers=2, output_size=length_size)
    elif model_type == "LSTM":
        net = LSTM(input_size=input_dim, hidden_size=66, num_layers=2, output_size=length_size)
    elif model_type == "LSTM_kan":
        net = LSTM_kan(input_size=input_dim, hidden_size=64, num_layers=2, output_size1= 72, output_size2 = length_size)
    elif model_type == "KAN":
        net = KAN(layers_hidden=[input_dim * window_size, length_size * output_dim], grid_size = 5, spline_order=3)
    elif model_type == "MLP":
        net = MLP(input_size = input_dim*window_size, hidden_size1 = 64, hidden_size2=32, output_size = length_size)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters in the {model_type} model: {total_params}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for t in range(1, epochs + 1):
        # Training loop
        net.train()
        for _, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            preds = net(datapoints) if model_type not in ["KAN", "MLP"] else net(datapoints.reshape(datapoints.shape[0], -1))
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, (datapoints, labels) in enumerate(dataloader_test):
                preds = net(datapoints) if model_type not in ["KAN", "MLP"] else net(datapoints.reshape(datapoints.shape[0], -1))
                val_loss += criterion(preds, labels).item()
        val_loss /= len(dataloader_test)

        print(f"Epoch {t}/{epochs}, Validation Loss: {val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = net.state_dict()  
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {t} epochs.")
            break

    # Save the best model
    best_model_path = f'model_data/{loaction_type}_{model_type}.pt'
    torch.save(best_model, best_model_path)

    training_time = time.time() - start_time
    print(f"Training time: {training_time}")

    return net, training_time

def model_test(x_data, y_data, length_size, model_type, scaler):
    if model_type == "GRU":
        net = GRU(input_size = input_dim, hidden_size = 64, num_layers = 2, output_size = length_size)
        net.load_state_dict(torch.load(f'model_data/{loaction_type}_GRU.pt', weights_only=False))  
    elif model_type == "LSTM":
        net = LSTM(input_size = input_dim, hidden_size = 66, num_layers = 2, output_size = length_size)
        net.load_state_dict(torch.load(f'model_data/{loaction_type}_LSTM.pt', weights_only=False))
    elif model_type == "LSTM_kan":
        net = LSTM_kan(input_size=input_dim, hidden_size = 64, num_layers=2, output_size1= 72, output_size2 = length_size)
        net.load_state_dict(torch.load(f'model_data/{loaction_type}_LSTM_kan.pt', weights_only=False))
    elif model_type == "KAN":
        net = KAN(layers_hidden=[input_dim*window_size, length_size*output_dim], grid_size=5, spline_order=3)
        net.load_state_dict(torch.load(f'model_data/{loaction_type}_KAN.pt', weights_only=False))
    elif model_type == "MLP":
        net = MLP(input_size = input_dim*window_size, hidden_size1 = 64, hidden_size2=32, output_size = length_size)
        net.load_state_dict(torch.load(f'model_data/{loaction_type}_MLP.pt', weights_only=False))
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    net.eval()
    if model_type == "KAN" or model_type == "MLP":
        pred = net(x_data.reshape(x_data.shape[0], -1))
    else:
        pred = net(x_data)

    pred = pred.detach().cpu()
    true = y_data.detach().cpu()
    pred_uninverse = scaler.inverse_transform(pred) 
    true_uninverse = scaler.inverse_transform(true)

    return pred_uninverse, true_uninverse

def generate_predictions(results_df, predictions, length_size = 168, front_nan=24):
    pred_list = []
    for i in range(0, len(predictions) - front_nan - 1, length_size):
        y_pred = predictions[i].flatten()
        pred_list.append(y_pred)
    # 创建 NaN 数组填充开头和结尾
    nan_array_front = np.full(front_nan, np.nan)
    nan_array_back = np.full(len(results_df) - len(pred_list) * length_size - front_nan, np.nan)
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
    mape = mean_absolute_percentage_error(actual_filtered, predicted_filtered)

    return mae, rmse, r2, mape

def plot_train_and_test(train_result, test_result):

    train_actual = train_result['Power']
    train_predicted = train_result['predict']
    test_actual = test_result['Power']
    test_predicted = test_result['predict']
    
    _, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(train_actual, label="Actual (Train)", color="blue", alpha=0.7, linewidth=1.5)
    axes[0].plot(train_predicted, label="Predicted (Train)", color="orange", alpha=0.7, linewidth=1.5)
    axes[0].set_title("Training Data: Actual vs Predicted")
    axes[0].set_ylabel("Power")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(test_actual, label="Actual (Test)", color="blue", alpha=0.7, linewidth=1.5)
    axes[1].plot(test_predicted, label="Predicted (Test)", color="orange", alpha=0.7, linewidth=1.5)
    axes[1].set_title("Testing Data: Actual vs Predicted")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Power")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_predictions(test_result):
    test_actual = test_result['Power']
    test_predicted = test_result['predict']

    absolute_errors = abs(test_actual - test_predicted)

    _, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    axes[0].plot(test_result['Time'], test_actual, label="True values", color="black", alpha=0.7, linewidth=1.5)
    axes[0].plot(test_result['Time'], test_predicted, label="Predicted values", color="blue", alpha=0.7, linewidth=1.5)
    axes[0].set_title("Load forecasting model performance degradation process diagram")
    axes[0].set_ylabel("Power")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(test_result['Time'], absolute_errors, label="Absolute errors", color="red", alpha=0.7, linewidth=1.5)
    axes[1].set_title("Absolute errors over time")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Power")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join('figure/prediction', f'{loaction_type}_{model_type}.png'))

if __name__ == '__main__':
    set_seed(42)
    data = pd.read_csv(f'data/{loaction_type}建模特征_3.csv')
    data = data.dropna()
    data = data.reset_index(drop = True)
    data['Time'] = pd.to_datetime(data['Time'])
    data['Hour'] = [np.cos(x * (2 * np.pi / 23)) for x in data['Hour']]

    larger_2018 = data[data['Time'] >= '2018-01-01 00:00:00'].index.min()
    smaller_2019 = data[data['Time'] < '2019-01-01 00:00:00'].index.max()
    larger_2019 = data[data['Time'] >= '2019-01-01 00:00:00'].index.min()
    smaller_2020 = data[data['Time'] < '2020-01-01 00:00:00'].index.max()

    train_result = data[:smaller_2019+1][['Time', 'Power']]
    validate_result = data[larger_2018:smaller_2019+1][['Time', 'Power']]
    test_result = data[larger_2019:smaller_2020+1][['Time', 'Power']]

    data_x = data[input_features_dict[loaction_type]]

    if (loaction_type != 'public') and (loaction_type != 'commercial'):
        data_x = data_x.drop(columns = ['Power', 'Hour', 'is_holiday'], axis = 1)
        no_need_scaler = data[['Hour', 'is_holiday']]
    else:
        data_x = data_x.drop(columns = ['Power', 'Hour'], axis = 1)
        no_need_scaler = data[['Hour']]

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    data_x = scaler_x.fit_transform(np.array(data_x))
    data_x = pd.DataFrame(data_x)
    data_x =  pd.concat([data_x, no_need_scaler], axis = 1)
    data_x = np.array(data_x)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    data_y = scaler_y.fit_transform(np.array(data[output_features]))

    '''
    data_x_numerical = data[[item for item in input_features_dict[loaction_type] if item not in ['Month', 'Hour', 'season']]]
    data_x_time = data[['Hour', 'Month']].copy()
    data_x_time['Hour'] = [np.cos(x * (2 * np.pi / 23)) for x in data_x_time['Hour']]
    data_x_time['Month'] = [np.sin(x * (2 * np.pi / 11)) for x in data_x_time['Month']]

    data_x_category = data[['season']] 
    encoder = OneHotEncoder(sparse_output = False)
    data_x_category = encoder.fit_transform(data_x_category[['season']])  

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    data_x_numerical = scaler_x.fit_transform(np.array(data_x_numerical))

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    data_y = scaler_y.fit_transform(np.array(data[output_features]))

    data_x = np.hstack([data_x_numerical, data_x_time.values, data_x_category])
    '''
    data_train_x = data_x[:smaller_2019+1, :]
    data_train_y = data_y[:smaller_2019+1, :]
    data_validate_x = data_x[larger_2018:smaller_2019+1, :]
    data_validate_y = data_y[larger_2018:smaller_2019+1, :]
    data_test_x = data_x[larger_2019:smaller_2020+1, :]
    data_test_y = data_y[larger_2019:smaller_2020+1, :]
    '''
    dataloader_train, X_train, y_train = data_loader(window_size, length_size, batch_size, data_train_x, data_train_y)
    dataloader_train, X_validate, y_validate = data_loader(window_size, length_size, batch_size, data_validate_x, data_validate_y)
    dataloader_test, X_test, y_test = data_loader(window_size, length_size, batch_size, data_test_x, data_test_y)

    _, training_time = model_train(model_type, patience = 5)
    y_validate_pred, y_validate_real = model_test(X_validate, y_validate, model_type, scaler_y)
    y_test_pred, y_test_real = model_test(X_test, y_test, model_type, scaler_y)
    train_result = generate_predictions(validate_result, y_validate_pred, length_size)
    test_result = generate_predictions(test_result, y_test_pred, length_size)

    train_mae, train_rmse, train_r2, train_mape = calculate_errors(train_result['Power'], train_result['predict'])
    test_mae, test_rmse, test_r2, test_mape = calculate_errors(test_result['Power'], test_result['predict'])
    test_result.to_csv(f'test_result/{loaction_type}_{model_type}_test_result.csv', index=False)
    
    score = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'MAE': [train_mae, test_mae],
            'RMSE': [train_rmse, test_rmse],
            'r2': [train_r2, test_r2],
            'MAPE': [train_mape, test_mape],
            'Training Time': [training_time, ' ']
    })

    score.to_csv((f"result/{loaction_type}_{model_type}.csv"), index=True)
    #plot_predictions(test_result)
    '''
    results = []
    for half_day in range(2, 62, 1):
        dataloader_train, X_train, y_train = data_loader(window_size, half_day*12, batch_size, data_train_x, data_train_y)
        dataloader_train, X_validate, y_validate = data_loader(window_size, half_day*12, batch_size, data_validate_x, data_validate_y)
        dataloader_test, X_test, y_test = data_loader(window_size, half_day*12, batch_size, data_test_x, data_test_y)

        _, training_time = model_train(model_type, half_day*12, patience = 8)
        y_validate_pred, y_validate_real = model_test(X_validate, y_validate, half_day*12, model_type, scaler_y)
        y_test_pred, y_test_real = model_test(X_test, y_test, half_day*12, model_type, scaler_y)
        train_result = generate_predictions(validate_result, y_validate_pred, half_day*12)
        test_result = generate_predictions(test_result, y_test_pred, half_day*12)

        train_mae, train_rmse, train_r2, train_mape = calculate_errors(train_result['Power'], train_result['predict'])
        test_mae, test_rmse, test_r2, test_mape = calculate_errors(test_result['Power'], test_result['predict'])
        #test_result.to_csv(f'test_result/{loaction_type}_{model_type}_test_result.csv', index=False)

        results.append({
             " ":0, 
            "Dataset": "Train",
            "MAE": train_mae,
            "RMSE": train_rmse,
            "r2": train_r2,
            "mape": train_mape,
            "prediction_step": half_day*12, 
            "Day": half_day*0.5,
        })

        results.append({
            " ":1, 
            "Dataset": "Test",
            "MAE": test_mae,
            "RMSE": test_rmse,
            "r2": test_r2,
            "mape": test_mape,
            "prediction_step": half_day*12, 
            "Day": half_day*0.5,
        })

    results = pd.DataFrame(results)
    results.to_csv(f"result/score_all/{loaction_type}_{model_type}_score_all.csv", index=False)