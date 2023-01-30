from torch import nn
import torch
import torchinfo

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNReLU6(nn.Sequential): # conv+bn+relu6; conv can be gconv
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU6, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, (kernel_size - 1) // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module): # expanded ConvBNReLU6 + (ConvBNReLU6) + expanded conv+bn; finally with a shortcut
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_shortcut = stride == 1 and in_channel == out_channel
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channel, in_channel * expand_ratio, kernel_size=1))
        layers.extend([
            ConvBNReLU6(in_channel * expand_ratio, in_channel * expand_ratio, stride=stride, groups=in_channel * expand_ratio),
            nn.Conv2d(in_channel * expand_ratio, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest) 
        last_channel = _make_divisible(1280 * alpha, round_nearest) 
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = []
        features.append(ConvBNReLU6(1, input_channel, stride=2)) # input channel = 1 for FashionMNIST dataset
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t)) 
                input_channel = output_channel
        features.append(ConvBNReLU6(input_channel, last_channel, 1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# test for shape
# net = MobileNetV2()
# report = torchinfo.summary(net, input_size=(1,1,224,224))
# summary_report = str(report)
# with open("MobileNetV2.txt", "w") as f:
#     f.write((summary_report))
