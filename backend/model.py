import torch
import torch.nn as nn

class LeNet(nn.Module):
    """
    Improved LeNet-5 for hand-drawn digit recognition.

    Key upgrades over the original:
      - MaxPool2d  instead of AvgPool2d  → sharper feature selection
      - BatchNorm  after each conv layer → faster convergence, more stable
      - Dropout(0.4) before final FC     → reduces overfitting to clean MNIST,
        generalises better to messy canvas input
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)        # 28→24
        self.bn1   = nn.BatchNorm2d(6)
        self.pool  = nn.MaxPool2d(2, 2)         # AvgPool → MaxPool
        self.conv2 = nn.Conv2d(6, 16, 5)        # 12→8
        self.bn2   = nn.BatchNorm2d(16)

        self.fc1     = nn.Linear(16 * 4 * 4, 120)
        self.fc2     = nn.Linear(120, 84)
        self.drop    = nn.Dropout(0.4)
        self.fc3     = nn.Linear(84, 10)

    def forward(self, x):
        activations = {}

        x = self.conv1(x)
        activations["conv1"] = x              # raw conv output (before BN/ReLU/pool)
        x = self.pool(torch.relu(self.bn1(x)))

        x = self.conv2(x)
        activations["conv2"] = x
        x = self.pool(torch.relu(self.bn2(x)))

        x = x.view(-1, 16 * 4 * 4)

        x = torch.relu(self.fc1(x))
        activations["fc1"] = x

        x = torch.relu(self.fc2(x))
        activations["fc2"] = x

        x = self.fc3(self.drop(x))

        return x, activations