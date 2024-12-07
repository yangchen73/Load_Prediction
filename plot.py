import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

def plot_predictions_10_days(test_results_dict, start_date, end_date):
    plt.figure(figsize=(16, 8))
    
    for model_name, df in test_results_dict.items():
        subset = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]
        if model_name == "Actual":
            plt.plot(subset['Time'], subset['Power'], label="Actual", color='black', linewidth=2)
        else:
            plt.plot(subset['Time'], subset['predict'], label=f"{model_name} predictions", linewidth=1.5)
    
    plt.title("Predicted vs Actual (10 Days, resident)", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Power", fontsize=14)
    plt.legend(loc="upper left", fontsize=8)
    ax = plt.gca() 
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

location_type = "office"

test_result_actual = pd.read_csv(f'test_result/{location_type}_LSTM_kan_test_result.csv')  # 替换路径和名称
test_result_lstm = pd.read_csv(f'test_result/{location_type}_LSTM_test_result.csv')  # 替换路径和名称
test_result_kan = pd.read_csv(f'test_result/{location_type}_KAN_test_result.csv')  # 替换路径和名称
test_result_mlp = pd.read_csv(f'test_result/{location_type}_MLP_test_result.csv')  # 替换路径和名称
test_result_lstm_t_kan = pd.read_csv(f'test_result/{location_type}_LSTM_kan_test_result.csv')  # 替换路径和名称
test_result_gru = pd.read_csv(f'test_result/{location_type}_GRU_test_result.csv')  # 替换路径和名称

test_results_dict = {
    'Actual': test_result_actual,
    'LSTM': test_result_lstm,
    'KAN': test_result_kan,
    'MLP': test_result_mlp,
    'LSTM-T-KAN': test_result_lstm_t_kan,
    'GRU': test_result_gru
}

start_date = "2019-04-20 00:00:00"  
end_date = "2019-04-30 00:00:00"    

plot_predictions_10_days(test_results_dict, start_date, end_date)
