import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

def plot_predictions_10_days(test_results_dict, start_date, end_date):
    plt.figure(figsize=(16, 8))
    plt.rc('font', family='Times New Roman')
    for model_name, df in test_results_dict.items():
        subset = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]
        subset.reset_index(drop=True, inplace=True)  # 重置索引，使横坐标从0开始
        if model_name == "Actual":
            plt.plot(subset.index, subset['Power'], label="Actual Value", color='black', linewidth=2)
        else:
            if model_name == "LSTM-T-KAN":
                plt.plot(subset.index, subset['predict'], label=f"{model_name}", color='red', linewidth=2)
            else:
                plt.plot(subset.index, subset['predict'], label=f"{model_name}", linewidth=1.5)

    plt.title("The actual value and predicted results of different models (office, 20 days)", fontsize=16)
    plt.xlabel("Time(Hour)", fontsize=16)  # 修改横坐标标签
    plt.ylabel("Load(kW)", fontsize=16)
    plt.legend(loc="upper left", fontsize=13)
    ax = plt.gca()
    ax.set_xlim(left=-10)  # 明确设置横坐标从0开始
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
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
    'GRU': test_result_gru,
    'LSTM-T-KAN': test_result_lstm_t_kan
}

start_date = "2019-08-10 00:00:00"  
end_date = "2019-08-30 00:00:00"    

plot_predictions_10_days(test_results_dict, start_date, end_date)
