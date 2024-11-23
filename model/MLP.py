import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.func1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.func2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.func1(x)
        out = self.relu(out)
        out = self.func2(out)
        out = self.sigmoid(out)
        return out