##Deep Learning HW2
##İpek Erdoğan
##2019700174
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.batch1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 3)
        self.batch2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 16, 3)
        self.batch3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 13 * 13,256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32,10)


    def forward(self, x):
        x = F.relu(self.batch1(self.conv1(x)))
        x = F.relu(self.batch2(self.conv2(x)))
        x = self.pool(F.relu(self.batch3(self.conv3(x))))
        flattens = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(flattens))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x,flattens

