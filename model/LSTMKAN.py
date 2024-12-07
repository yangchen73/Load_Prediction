import torch
import torch.nn as nn
from model.KAN import KAN
    
class LSTM_kan(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size1, output_size2, dropout = 0.2):
        super(LSTM_kan, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0  
        )
        #self.func = nn.Linear(hidden_size, output_size1)
        #self.dropout = nn.Dropout(dropout) 
        self.kan = KAN([hidden_size, output_size2])

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        #out = self.func(out[:, -1, :])
        #out = self.dropout(out[:, -1, :]) 
        out = self.kan(out[:, -1, :])
        return out

