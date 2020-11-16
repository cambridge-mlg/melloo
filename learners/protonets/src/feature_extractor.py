import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, in_channels):
        self.in_channels = in_channels
        super(ConvNet, self).__init__()
        self.encoder = nn.Sequential(
            self._make_conv2d_layer(self.in_channels, 64),
            self._make_conv2d_layer(64, 64),
            self._make_conv2d_layer(64, 64),
            self._make_conv2d_layer(64, 64)
        )

    def _make_conv2d_layer(self, in_maps, out_channels):
        bn = nn.BatchNorm2d(out_channels)
        nn.init.uniform_(bn.weight)  # for pytorch 1.2 or later
        return nn.Sequential(
            nn.Conv2d(in_maps, out_channels, kernel_size=3, stride=1, padding=1),
            bn,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # flatten
        return x

    @property
    def output_size(self):
        if self.in_channels == 3:
            return 1600
        else:
            return 64


class BottleneckNet(nn.Module):
    def __init__(self, in_channels):
        self.in_channels = in_channels
        super(BottleneckNet, self).__init__()
        self.encoder = nn.Sequential(
            self._make_conv2d_layer(self.in_channels, 64),
            self._make_conv2d_layer(64, 64),
            self._make_conv2d_layer(64, 64),
            self._make_conv2d_layer(64, 64),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(in_features=64, out_features=2, bias=True)


    def _make_conv2d_layer(self, in_maps, out_channels):
        bn = nn.BatchNorm2d(out_channels)
        nn.init.uniform_(bn.weight)  # for pytorch 1.2 or later
        return nn.Sequential(
            nn.Conv2d(in_maps, out_channels, kernel_size=3, stride=1, padding=1),
            bn,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

    @property
    def output_size(self):
        return 2


