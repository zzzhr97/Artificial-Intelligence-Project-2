import torch
import torch.nn as nn
from network import SimpleNet, SimpleResNet

_model = SimpleResNet(4)

class Net(nn.Module):
    def __init__(self, model=_model):
        super(Net, self).__init__()
        self.model = model
        self.out_channel = 64

        # policy part
        self.policy_layers = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=self.out_channel, out_channels=16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=16 * 15 * 15, out_features=15 * 15),
            nn.LogSoftmax(dim=1)
        )

        # value part
        self.value_layers = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=self.out_channel, out_channels=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=4 * 15 * 15, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.Tanh()
        )

    def forward(self, state):
        x = self.model(state)

        # policy part
        policy = self.policy_layers(x).view(-1, 15, 15)

        # value part
        value = self.value_layers(x)

        assert policy.shape == (state.shape[0], 15, 15) and value.shape == (state.shape[0], 1), \
            "policy.shape: {}, value.shape: {}".format(policy.shape, value.shape)

        return policy, value