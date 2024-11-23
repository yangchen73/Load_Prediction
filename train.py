import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import cycle
from model.LSTMKAN import LSTM_kan
from model.GRU import GRU
from model.LSTM import LSTM
from model.KAN import KAN
from model.MLP import MLP
from sklearn.preprocessing import MinMaxScaler

loaction_type = 'resident'
model_type = 'GRU'

input_features_office = ['Power', 'max_power', 'power_t-1', 'min_power','RMS','10%_quantile_power','power_t-2','min_power','Peak_to_Peak','power_t-12','power_t-4','power_t-10', 'power_t-8', 'power_t-7' ,'approx_max' ,'power_t-3' ,'90%_quantile_power' ,'Temperature', 'Duration_Above90%', 'power_t-11', 'Duration_Below10%', 'power_t-5', 'detail_kurtosis', 'median_power', 'power_t-6', 'std_power', 'Humidity', 'approx_sd', 'FFT_Magnitude_mean', 'approx_energy', 'Month', 'Hour', 'is_holiday']

input_features_commercial = ['Power','max_power','RMS','min_power','10%_quantile_power','power_t-2','power_t-9','90%_quantile_power','power_t-12','approx_energy','power_t-1','power_t-4','power_t-3','Peak_to_Peak','FFT_Total_energy','power_t-11','approx_max','FFT_Magnitude_max','power_t-8','Duration_Below10%','power_t-7','Duration_Above90%','std_power','power_t-6','Temperature','approx_sd','power_t-10','FFT_Magnitude_mean','power_t-5','Humidity','Month','Hour']

input_features_public = ['Power', 'max_power','power_t-1','90%_quantile_power','Peak_to_Peak','10%_quantile_power','min_power','power_t-12','power_t-2','power_t-9','power_t-8','Month','Hour']

input_features_resident = ['Power','RMS','median_power','90%_quantile_power','power_t-1','max_power','10%_quantile_power','approx_mean','power_t-2','min_power','approx_energy','FFT_Magnitude_max','power_t-12','FFT_Total_energy','Duration_Below10%','power_t-9','approx_max','Temperature','power_t-8','power_t-4','power_t-3','Humidity','power_t-5','Duration_Above90%','Peak_to_Peak','power_t-11','power_t-6','power_t-7','Month','Hour']

input_features_dict = {'office': input_features_office, 'commercial': input_features_commercial, 'public': input_features_public,'resident': input_features_resident}

output_features = ['Power']
train_ratio = 0.8
window_size = 24
input_dim = len(input_features_dict[loaction_type])
output_dim = len(output_features)
length_size = 168
batch_size = 128
epochs = 60

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

def model_train(model_type):
    start_time = time.time()
    if model_type == "GRU":
        net = GRU(input_size = input_dim, hidden_size = 50, num_layers = 2, output_size = length_size)
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
        best_model_path = f'model_data/{loaction_type}_GRU.pt'
        torch.save(net.state_dict(), best_model_path)

    elif model_type == "LSTM":
        net = LSTM(input_size = input_dim, hidden_size = 50, num_layers = 2, output_size = length_size)
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
        torch.save(net.state_dict(), f'model_data/{loaction_type}_LSTM.pt')

    elif model_type == "LSTM_kan":
        net = LSTM_kan(input_size = input_dim, hidden_size = 50, num_layers = 2, output_size = length_size)
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
        torch.save(net.state_dict(), f'model_data/{loaction_type}_LSTM_kan.pt')

    elif model_type == "KAN":
        net = KAN(layers_hidden=[input_dim*window_size, length_size*output_dim], grid_size=5, spline_order=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        for t in range(1, epochs + 1): 
            for _, (datapoints, labels) in enumerate(dataloader_train):
                optimizer.zero_grad()
                datapoints = datapoints.reshape(datapoints.shape[0], -1)
                preds = net(datapoints)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {t}/{epochs}")
        torch.save(net.state_dict(), f'model_data/{loaction_type}_KAN.pt')

    elif model_type == "MLP":
        net = MLP(input_size = input_dim*window_size, hidden_size = 50, output_size = length_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        for t in range(1, epochs + 1): 
            for _, (datapoints, labels) in enumerate(dataloader_train):
                optimizer.zero_grad()
                datapoints = datapoints.reshape(datapoints.shape[0], -1)
                preds = net(datapoints)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {t}/{epochs}")
        torch.save(net.state_dict(), f'model_data/{loaction_type}_MLP.pt')

    training_time = time.time() - start_time
    print(f"Training time: {training_time}")

    return net

def model_test(x_data, y_data, model_type, scaler):
    if model_type == "GRU":
        net = GRU(input_size = input_dim, hidden_size = 50, num_layers = 2, output_size = length_size)
        net.load_state_dict(torch.load(f'model_data/{loaction_type}_GRU.pt', weights_only=False))  
    elif model_type == "LSTM":
        net = LSTM(input_size = input_dim, hidden_size = 50, num_layers = 2, output_size = length_size)
        net.load_state_dict(torch.load(f'model_data/{loaction_type}_LSTM.pt', weights_only=False))
    elif model_type == "LSTM_kan":
        net = LSTM_kan(input_size = input_dim, hidden_size = 50, num_layers = 2, output_size = length_size)
        net.load_state_dict(torch.load(f'model_data/{loaction_type}_LSTM_kan.pt', weights_only=False))
    elif model_type == "KAN":
        net = KAN(layers_hidden=[input_dim*window_size, length_size*output_dim], grid_size=5, spline_order=3)
        net.load_state_dict(torch.load(f'model_data/{loaction_type}_KAN.pt', weights_only=False))
    elif model_type == "MLP":
        net = MLP(input_size = input_dim*window_size, hidden_size = 50, output_size = length_size)
        net.load_state_dict(torch.load(f'model_data/{loaction_type}_MLP.pt', weights_only=False))

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
    return mae, rmse, r2

def plot_predictions(train_result, test_result):

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

if __name__ == '__main__':
    data = pd.read_csv(f'data/{loaction_type}建模特征_1.csv')
    data = data.dropna()
    data = data.reset_index(drop = True)
    train_result = data[:int(train_ratio*len(data))][['Power']]
    test_result = data[int(train_ratio*len(data)):][['Power']]
    data_x = data[input_features_dict[loaction_type]]
    data_y = data[output_features]
    scaler_x = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    data_x = scaler_x.fit_transform(np.array(data_x))
    data_y = scaler_y.fit_transform(np.array(data_y))
    data_train_x = data_x[:int(train_ratio*len(data)), :]
    data_train_y = data_y[:int(train_ratio*len(data)), :]
    data_test_x = data_x[int(train_ratio*len(data)):, :]
    data_test_y = data_y[int(train_ratio*len(data)):, :]

    dataloader_train, X_train, y_train = data_loader(window_size, length_size, batch_size, data_train_x, data_train_y)
    dataloader_test, X_test, y_test = data_loader(window_size, length_size, batch_size, data_test_x, data_test_y)

    model_train(model_type)
    y_train_pred, y_train_real = model_test(X_train, y_train, model_type, scaler_y)
    y_test_pred, y_test_real = model_test(X_test, y_test, model_type, scaler_y)
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


    score.to_csv((f"result/{loaction_type}_{model_type}.csv"), index=True)
    plot_predictions(train_result, test_result)