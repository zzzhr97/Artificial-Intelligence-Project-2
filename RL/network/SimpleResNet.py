import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_depth, output_depth):
        super(ResidualBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_depth, output_depth, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_depth),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(output_depth, output_depth, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_depth),
            nn.ReLU()
        )

    def forward(self, x):
        x_0 = x
        x = self.block1(x)
        x = self.block2(x)

        if x_0.shape[1] == x.shape[1]:
            x = x + x_0     # 不能使用 x += x_0!

        return x
    
class SimpleResNet(nn.Module):
    def __init__(self, in_channel):
        super(SimpleResNet, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(in_channel, 8),       # (4, 15, 15) -> (8, 15, 15)
                ResidualBlock(8, 8),                # (8, 15, 15) -> (8, 15, 15)
            ),
            nn.Sequential(
                ResidualBlock(8, 16),               # (8, 15, 15) -> (16, 15, 15)
                ResidualBlock(16, 16),              # (16, 15, 15) -> (16, 15, 15)
            ),
            nn.Sequential(
                ResidualBlock(16, 32),              # (16, 15, 15) -> (32, 15, 15)
                ResidualBlock(32, 32),              # (32, 15, 15) -> (32, 15, 15)
            ),
            nn.Sequential(
                ResidualBlock(32, 64),              # (32, 15, 15) -> (64, 15, 15)
                ResidualBlock(64, 64),              # (64, 15, 15) -> (64, 15, 15)
            ),
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        assert x.shape == (x.shape[0], 64, 15, 15), x.shape
        return x