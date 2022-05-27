import torch
import torch.nn as nn

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepmindNet(nn.Module):
    def __init__(self):
        super(DeepmindNet, self).__init__()

        self._conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=8,
                stride=4,
                device=device
            ),
            nn.BatchNorm2d(32, device=device),
            nn.ReLU()
        )

        self._conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                device=device
            ),
            nn.BatchNorm2d(64, device=device),
            nn.ReLU()
        )

        self._conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                device=device
            ),
            nn.BatchNorm2d(64, device=device),
            nn.ReLU()
        )

        self._ln1 = nn.Sequential(
            nn.Linear(3136, 512, device=device),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self._ln2 = nn.Sequential(
            nn.Linear(512, 9, device=device),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.to(device)

        x = self._conv1(x)

        x = self._conv2(x)

        x = self._conv3(x)

        x = x.view(-1) if x.dim() == 3 else x.view(x.shape[0], -1)

        x = self._ln1(x)
        x = self._ln2(x)

        return x


class Alexnet(nn.Module):

    def __init__(self):
        super(Alexnet, self).__init__()

        self._conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding=0,
                device=device),
            nn.ReLU(),
        )

        self._mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self._norm1 = nn.BatchNorm2d(96, device=device)

        self._conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=5,
                      padding=2,
                      device=device),
            nn.ReLU()
        )

        self._mp2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self._norm2 = nn.BatchNorm2d(256, device=device)

        self._conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=384,
                      kernel_size=3,
                      padding=1,
                      device=device),
            nn.ReLU()
        )

        self._conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=384,
                      kernel_size=3,
                      padding=1,
                      device=device),
            nn.ReLU()
        )

        self._conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      padding=1,
                      device=device),
            nn.ReLU()
        )

        self._mp3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self._ln1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096, device=device),
            nn.ReLU()
        )

        self._ln2 = nn.Sequential(
            nn.Linear(4096, 4096, device=device),
            nn.ReLU()
        )

        self._ln3 = nn.Linear(4096, 9,
                              device=device)

    def forward(self, x):
        x = x.to(device)
        x = self._conv1(x)
        x = self._mp1(x)
        x = self._norm1(x)

        x = self._conv2(x)
        x = self._mp2(x)
        x = self._norm2(x)

        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)

        x = self._mp3(x)

        x = x.view(-1) if x.dim() == 3 else x.view(x.shape[0], -1)

        x = self._ln1(x)
        x = F.dropout(x, 0.5)
        x = self._ln2(x)
        x = F.dropout(x, 0.5)
        x = self._ln3(x)

        return x
