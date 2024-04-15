import torch.nn as nn


# Define the model
class CNN2D(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=25,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=25,
            padding="same",
        )
        self.fc1 = nn.Linear(88 * 64, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3, ceil_mode=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.reshape((-1, 1, 784))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x)) * 10
        return x
