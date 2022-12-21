import torch.nn as nn


class ConvnetFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(ConvnetFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            self._make_conv2d_layer(self.in_channels, self.out_channels),
            self._make_conv2d_layer(self.out_channels, self.out_channels),
            self._make_conv2d_layer(self.out_channels, self.out_channels),
            self._make_conv2d_layer(self.out_channels, self.out_channels)
        )

    def _make_conv2d_layer(self, in_channels, out_channels):
        bn = nn.BatchNorm2d(out_channels)
        nn.init.uniform_(bn.weight)  # for pytorch 1.2 or later
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
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
        if self.out_channels == 64:
            return 1600
        else:
            return 800


class ProtoNets(nn.Module):
    def __init__(self, num_classes=64):
        super(ProtoNets, self).__init__()
        self.feature_extractor = ConvnetFeatureExtractor(in_channels=3, out_channels=64)
        self.fc = nn.Linear(in_features=self.feature_extractor.output_size, out_features=num_classes)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.fc(out)
        return out


class MamlNet(nn.Module):
    def __init__(self, num_classes=64):
        super(MamlNet, self).__init__()
        self.feature_extractor = ConvnetFeatureExtractor(in_channels=3, out_channels=32)
        self.fc = nn.Linear(in_features=self.feature_extractor.output_size, out_features=num_classes)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.fc(out)
        return out
