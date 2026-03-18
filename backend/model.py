import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        activations = {}

        x = self.conv1(x)
        activations["conv1"] = x
        x = self.pool(torch.relu(x))

        x = self.conv2(x)
        activations["conv2"] = x
        x = self.pool(torch.relu(x))

        x = x.view(-1, 16 * 4 * 4)

        x = torch.relu(self.fc1(x))
        activations["fc1"] = x

        x = torch.relu(self.fc2(x))
        activations["fc2"] = x

        x = self.fc3(x)

        return x, activations