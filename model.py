import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class AlexNet(nn.Module):
    def __init__(self, class_number):
        super(AlexNet, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fclayer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4096),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),

            nn.Linear(in_features=4096, out_features=class_number)
        )

    def forward(self, x):
        x = self.convlayer(x)
        x = x.view(-1, 9216)
        x = self.fclayer(x)
        return F.log_softmax(x, dim=1)