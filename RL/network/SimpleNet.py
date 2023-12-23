import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_channel):
        super(SimpleNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        assert x.shape == (x.shape[0], 64, 15, 15), x.shape
        return x