import torch.nn as nn


class RotateClassifier(nn.Module):
    def __init__(self):
        super(RotateClassifier, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.LeakyReLU(.2),
            nn.MaxPool2d(2),

        )
        self.con2 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.LeakyReLU(.2),
            nn.MaxPool2d(2),
        )
        self.con3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(.2),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*28*28, 32),
            nn.LeakyReLU(.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x = self.con3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
