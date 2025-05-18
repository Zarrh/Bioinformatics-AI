import torch.nn as nn


class BrainTumorCNN(nn.Module):
  def __init__(self, num_classes=4):
    super(BrainTumorCNN, self).__init__()

    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.dropout = nn.Dropout(0.5)

    # Conv Block 1
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # (B, 32, 224, 224)

    # Conv Block 2
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (B, 64, 112, 112)

    # Conv Block 3
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (B, 128, 56, 56)

    # Conv Block 4
    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # (B, 256, 28, 28)

    # Conv Block 5
    self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # (B, 256, 14, 14)

    # Final shape after pool: (B, 256, 7, 7)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(256 * 7 * 7, 512)
    # self.fc1 = nn.Linear(32 * 112 * 112, 512)
    self.fc2 = nn.Linear(512, num_classes)

  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))  # (B, 32, 112, 112)
    x = self.pool(self.relu(self.conv2(x)))  # (B, 64, 56, 56)
    x = self.pool(self.relu(self.conv3(x)))  # (B, 128, 28, 28)
    x = self.pool(self.relu(self.conv4(x)))  # (B, 256, 14, 14)
    x = self.pool(self.relu(self.conv5(x)))  # (B, 256, 7, 7)

    x = self.flatten(x)
    x = self.dropout(self.relu(self.fc1(x)))
    x = self.fc2(x)

    return x