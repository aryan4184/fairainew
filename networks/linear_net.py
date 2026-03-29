import torch
import torch.nn as nn


class LinearNetSig(nn.Module):
    """
    Linear binary classifier with sigmoid output.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))
    

class LinearNetDefer(nn.Module):
    """
    Linear classifier with (K + 1) outputs.
    The extra unit represents the deferral / reject option.
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes + 1)

    def forward(self, x):
        return self.fc(x)
    
class LinearNet(nn.Module):
    """
    Standard linear classifier with K outputs (no softmax).
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    