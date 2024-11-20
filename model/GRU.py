import torch.nn as nn
    
class GRU(nn.Module):
    def __init__(self, n_hidden, n_features, length_size):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=n_hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=n_hidden, out_features=length_size)

    def forward(self, x):
        out = self.gru(x)[0][:, -1, :]
        out = self.fc(out)
        out = out.squeeze()
        return out