import torch.nn.functional as F
from torch import nn
import torch


class FullyConnectedNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(2304, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.final = nn.Linear(32, 7)

        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):

        x = self.flatten(x)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.final(x)

        return x


class SimpleConvNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Conv2d(1, 32, 3)

        self.layer2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(start_dim=1)
        self.final_layer = nn.Linear(6400, 7)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = self.pool(x)

        x = F.relu(self.layer2(x))
        x = self.pool(x)

        x = self.flatten(x)
        return self.final_layer(x)


class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # The input size for fc1 is now adjusted for 48x48 images after 3 conv + pool layers
        # Input size to fc1: 128 filters * (48 / 2 / 2 / 2)^2 = 128 * 6 * 6 = 4608
        self.fc1 = nn.Linear(in_features=128 * 6 * 6, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

        self.flatten = nn.Flatten(start_dim=1)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))

        x = self.flatten(x)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class EDA_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Conv2d(1, 32, 3)

        self.layer2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(6400, 64)
        self.dense2 = nn.Linear(64, 7)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = self.pool(x)

        x = F.relu(self.layer2(x))
        x = self.pool(x)

        x = self.flatten(x)

        x = F.relu(self.dense1(x))
        x = F.softmax(self.dense2(x))
        return x
