import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import cycle
from model.LSTMKAN import LSTM_kan
from model.GRU import GRU
from model.LSTM import LSTM
from model.KAN import KAN
from sklearn.preprocessing import MinMaxScaler

input_features = ['Power', 'max_power', 'power_t-1', 'min_power','RMS','10%_quantile_power','power_t-2','min_power','Peak_to_Peak','power_t-12','power_t-4','power_t-10', 'power_t-8', 'power_t-7' ,'approx_max' ,'power_t-3' ,'90%_quantile_power' ,'Temperature', 'Duration_Above90%', 'power_t-11', 'Duration_Below10%', 'power_t-5', 'detail_kurtosis', 'median_power', 'power_t-6', 'std_power', 'Humidity', 'approx_sd', 'FFT_Magnitude_mean', 'approx_energy', 'Month', 'Hour', 'is_holiday']

output_features = ['Power']
input_dim = len(input_features)
output_dim = len(output_features)
train_ratio = 0.8
window_size = 24
length_size = 168
batch_size = 64
epochs = 100
model_type = 'LSTM'

def data_loader(window, length_size, batch_size, data):
    seq_len = window  
    sequence_length = seq_len + length_size  
    result = []  
    for index in range(len(data) - sequence_length):  
        result.append(data[index: index + sequence_length]) 
    result = np.array(result)  
    x_data = result[:, :-length_size] 
    y_data = result[:, -length_size:, -1]  
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], input_dim))  
    y_data = np.reshape(y_data, (y_data.shape[0], -1))

    x_data, y_data = torch.tensor(x_data).to(torch.float32), torch.tensor(y_data).to(
        torch.float32) 
    ds = torch.utils.data.TensorDataset(x_data, y_data)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,shuffle=True) 
    return dataloader, x_data, y_data

def model_train(model_type):
    start_time = time.time()
    if model_type == "GRU":
        net = GRU(n_features = input_dim, n_hidden = 5, length_size = length_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        for t in range(1, epochs + 1): 
            for _, (datapoints, labels) in enumerate(dataloader_train):
                optimizer.zero_grad()
                preds = net(datapoints)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {t}/{epochs}")
        best_model_path = 'model_data/GRU.pt'
        torch.save(net.state_dict(), best_model_path)

    elif model_type == "LSTM":
        net = LSTM(n_features = input_dim, n_hidden = 5, length_size = length_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        for t in range(1, epochs + 1): 
            for _, (datapoints, labels) in enumerate(dataloader_train):
                optimizer.zero_grad()
                preds = net(datapoints)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {t}/{epochs}")
        torch.save(net.state_dict(), 'model_data/LSTM.pt')

    elif model_type == "LSTM_kan":
        net = LSTM_kan(input_dim = input_dim, hidden_dim = 32, num_layers = 2, output_dim = output_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        for t in range(1, epochs + 1): 
            for _, (datapoints, labels) in enumerate(dataloader_train):
                optimizer.zero_grad()
                preds = net(datapoints)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {t}/{epochs}")
        torch.save(net.state_dict(), 'model_data/LSTM_kan.pt')

    elif model_type == "KAN":
        net = KAN(layers_hidden=[input_dim, length_size*output_dim], grid_size=5, spline_order=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        for t in range(1, epochs + 1): 
            for _, (datapoints, labels) in enumerate(dataloader_train):
                optimizer.zero_grad()
                preds = net(datapoints[:, -1, :])
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {t}/{epochs}")
        torch.save(net.state_dict(), 'model_data/KAN.pt')

    training_time = time.time() - start_time
    print(f"Training time: {training_time}")

    return net

def model_test(x_data, y_data, model_type):
    if model_type == "GRU":
        net = GRU(n_features = input_dim,  n_hidden = 5, length_size = length_size)
        net.load_state_dict(torch.load('model_data/GRU.pt', weights_only=False))  
    elif model_type == "LSTM":
        net = LSTM(n_features = input_dim,  n_hidden = 5, length_size = length_size)
        net.load_state_dict(torch.load('model_data/LSTM.pt', weights_only=False))
    elif model_type == "LSTM_kan":
        net = LSTM_kan(input_dim = input_dim, hidden_dim = 32, num_layers = 2, output_dim = output_dim)
        net.load_state_dict(torch.load('model_data/LSTM_kan.pt', weights_only=False))
    elif model_type == "KAN":
        net = KAN(layers_hidden=[input_dim, length_size*output_dim], grid_size=5, spline_order=3)
        net.load_state_dict(torch.load('model_data/KAN.pt', weights_only=False))

    net.eval()
    if model_type == "KAN":
        pred = net(x_data[:, -1, :])
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
    print(actual_filtered)
    predicted_filtered = predicted[mask]
    print(predicted_filtered)
    
    mae = mean_absolute_error(actual_filtered, predicted_filtered)
    rmse = np.sqrt(mean_squared_error(actual_filtered, predicted_filtered))
    r2 = r2_score(actual_filtered, predicted_filtered)
    return mae, rmse, r2

if __name__ == '__main__':
    data = pd.read_csv('data/office建模特征_1.csv')
    data = data.dropna()
    data = data.reset_index(drop = True)
    train_result = data[:int(train_ratio*len(data))][['Power']]
    test_result = data[int(train_ratio*len(data)):][['Power']]
    data = data[input_features]
    data_target = data[output_features]
    scaler = MinMaxScaler(feature_range=(0,1))
    data_inverse = scaler.fit_transform(np.array(data))
    data = data_inverse
    data_train = data[:int(train_ratio*len(data)), :]
    data_test = data[int(train_ratio*len(data)):, :]

    dataloader_train, X_train, y_train = data_loader(window_size, length_size, batch_size, data_train)
    dataloader_test, X_test, y_test = data_loader(window_size, length_size, batch_size, data_test)

    #model_train(model_type)
    data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
    data_train_uninverse = scaler.inverse_transform(np.array(data_train).reshape(-1, 1))
    y_train_pred, y_train_real = model_test(X_train, y_train, model_type)
    y_test_pred, y_test_real = model_test(X_test, y_test, model_type)
    train_result = generate_predictions(train_result, y_train_pred)
    test_result = generate_predictions(test_result, y_test_pred)

    train_mae, train_rmse, train_r2 = calculate_errors(train_result['Power'], train_result['predict'])
    test_mae, test_rmse, test_r2 = calculate_errors(test_result['Power'], test_result['predict'])
    score = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'MAE': [train_mae, test_mae],
            'RMSE': [train_rmse, test_rmse],
            'r2': [train_r2, test_r2]
    })

    score.to_csv(("result/LSTM.csv"), index=True)
