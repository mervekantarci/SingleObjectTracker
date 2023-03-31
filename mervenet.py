""" Merve Gul Kantarci Vision Lab Assignment 3"""

from torch import nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # fc relu dropout fc relu droupout fc layers
        self.fc1 = nn.Linear(1024, 1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.drp1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.drp2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 4)

    def forward(self, x):
        # forward through network
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drp1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drp2(x)
        x = self.fc3(x)
        return x
