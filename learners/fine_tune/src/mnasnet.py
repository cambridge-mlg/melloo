import warnings

import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['MNASNet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']

_MODEL_URLS = {
    "mnasnet0_5":
    "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
    "mnasnet0_75": None,
    "mnasnet1_0":
    "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
    "mnasnet1_3": None
}

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor, bn_fn, bn_momentum=0.1):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            bn_fn(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=mid_ch, bias=False),
            bn_fn(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            bn_fn(out_ch, momentum=bn_momentum))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def film(x, gamma, beta):
    gamma = gamma[None, :, None, None]
    beta = beta[None, :, None, None]
    return gamma * x + beta


class _FilmInvertedResidual(_InvertedResidual):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor, bn_fn, bn_momentum=0.1):
        _InvertedResidual.__init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor, bn_fn, bn_momentum)

    def forward(self, input, gamma, beta):
        # execute the first 5 layers normally
        # (0): Conv2d
        # (1): bn_fn
        # (2): ReLU
        # (3): Conv2d
        # (4): bn_fn
        x = input
        for layer in range(5):
            x = self.layers[layer](x)

        # film
        x = film(x, gamma, beta)

        # execute the final 3 layers normally
        for layer in range(5, 8):
            x = self.layers[layer](x)

        if self.apply_residual:
            return x + input
        else:
            return x


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats, bn_momentum, inverted_residual_fn, bn_fn):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = inverted_residual_fn(in_ch, out_ch, kernel_size, stride, exp_factor, bn_fn, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            inverted_residual_fn(out_ch, out_ch, kernel_size, 1, exp_factor, bn_fn, bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, alpha, bn_fn, inverted_residual_fn=_InvertedResidual, num_classes=1000, dropout=0.2):
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False),
            bn_fn(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1,
                      groups=depths[0], bias=False),
            bn_fn(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False),
            bn_fn(depths[1], momentum=_BN_MOMENTUM),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM, inverted_residual_fn, bn_fn),
            _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM, inverted_residual_fn, bn_fn),
            _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM, inverted_residual_fn, bn_fn),
            _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM, inverted_residual_fn, bn_fn),
            _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM, inverted_residual_fn, bn_fn),
            _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM, inverted_residual_fn, bn_fn),
            # Final mapping to classifier input.
            nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
            bn_fn(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        # self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
        #                                 nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x, film_param_dict=None):
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                         nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get("version", None)
        assert version in [1, 2]

        if version == 1 and not self.alpha == 1.0:
            # In the initial version of the model (v1), stem was fixed-size.
            # All other layer configurations were the same. This will patch
            # the model so that it's identical to v1. Model with alpha 1.0 is
            # unaffected.
            depths = _get_depths(self.alpha)
            v1_stem = [
                nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32,
                          bias=False),
                nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False),
                nn.BatchNorm2d(16, momentum=_BN_MOMENTUM),
                _stack(16, depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
            ]
            for idx, layer in enumerate(v1_stem):
                self.layers[idx] = layer

            # The model is now identical to v1, and must be saved as such.
            self._version = 1
            warnings.warn(
                "A new version of MNASNet model has been implemented. "
                "Your checkpoint was saved using the previous version. "
                "This checkpoint will load and work as before, but "
                "you may want to upgrade by training a newer model or "
                "transfer learning from an updated ImageNet checkpoint.",
                UserWarning)

        super(MNASNet, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    @property
    def output_size(self):
        return 1280


class FilmMNASNet(MNASNet):
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, alpha, bn_fn, inverted_residual_fn):
        MNASNet.__init__(self, alpha=alpha, bn_fn=bn_fn, inverted_residual_fn=inverted_residual_fn, num_classes=1000, dropout=0.2)

    def forward(self, x, film_param_dict):
        x = self.execute_layers(x, film_param_dict)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return x

    def execute_layers(self, x, film_param_dict):
        # execute the first 2 layers
        # (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # (1): BatchNorm2d(32, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        for layer in range(0, 2):
            x = self.layers[layer](x)

        # FiLM
        x = film(x, film_param_dict[0][0]['gamma'], film_param_dict[0][0]['beta'])

        # execute the next 3 layers
        # (2): ReLU(inplace=True)
        # (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        # (4): BatchNorm2d(32, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        for layer in range(2, 5):
            x = self.layers[layer](x)

        # FiLM
        x = film(x, film_param_dict[0][1]['gamma'], film_param_dict[0][1]['beta'])

        # execute the next 3 layers
        # (5): ReLU(inplace=True)
        # (6): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # (7): BatchNorm2d(16, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        for layer in range(5, 8):
            x = self.layers[layer](x)

        # FiLM
        x = film(x, film_param_dict[0][2]['gamma'], film_param_dict[0][2]['beta'])

        # The next 6 layers 8 - 13 inclusive are Sequentials of _FilmInvertedResidual
        # We need to pass in the film layer coefficients.
        start_index = 8
        for layer in range(start_index, 14):
            layer_index = layer - start_index + 1 # + 1 due to the FiLM layers before the FilmInvertedResidual
            for stack in range(len(self.layers[layer])):
                x = self.layers[layer][stack](x,
                                              film_param_dict[layer_index][stack]['gamma'],
                                              film_param_dict[layer_index][stack]['beta'])

        # execute next 2 layers
        # (14): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # (15): BatchNorm2d(1280, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        for layer in range(14, 16):
            x = self.layers[layer](x)

        # FiLM
        x = film(x, film_param_dict[-1][0]['gamma'], film_param_dict[-1][0]['beta'])

        # (16): ReLU(inplace=True)
        x = self.layers[16](x)

        return x


def _load_pretrained(model_name, model, progress):
    if model_name not in _MODEL_URLS or _MODEL_URLS[model_name] is None:
        raise ValueError(
            "No checkpoint is available for model type {}".format(model_name))
    checkpoint_url = _MODEL_URLS[model_name]
    model.load_state_dict(
        load_state_dict_from_url(checkpoint_url, progress=progress), strict=False)


def mnasnet0_5(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.5, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet0_5", model, progress)
    return model


def mnasnet0_75(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.75, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet0_75", model, progress)
    return model


def mnasnet1_0(pretrained=False, progress=True, pretrained_model_path=None, batch_normalization='eval', **kwargs):
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    bn_fn = nn.BatchNorm2d

    model = MNASNet(alpha=1.0, bn_fn=bn_fn, **kwargs)
    if pretrained:
        if pretrained_model_path == 'built-in':
            _load_pretrained("mnasnet1_0", model, progress)
        else:
            model.load_state_dict(torch.load(pretrained_model_path))
    return model


def film_mnasnet1_0(pretrained=False, progress=True, pretrained_model_path=None, batch_normalization='eval', **kwargs):
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    bn_fn = nn.BatchNorm2d

    model = FilmMNASNet(alpha=1.0, inverted_residual_fn=_FilmInvertedResidual, bn_fn=bn_fn, **kwargs)
    if pretrained:
        if pretrained_model_path == 'built-in':
            _load_pretrained("mnasnet1_0", model, progress)
        else:
            model.load_state_dict(torch.load(pretrained_model_path))
    return model




def mnasnet1_3(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.3, **kwargs)
    if pretrained:
        _load_pretrained("mnasnet1_3", model, progress)
    return model
