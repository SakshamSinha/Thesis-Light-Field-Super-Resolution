from torch import nn


class Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(12, 64, 9, stride=1, padding=4)
        self.relu1= nn.ReLU(True)
        self.conv2 = nn.Conv2d(64, 3, 9, stride=1, padding=1)
        self.relu2 = nn.ReLU(True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu2(y)
        return (y)