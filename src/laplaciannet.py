import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = LayerNorm
        self.conv1 = conv3x3(inplanes, planes, stride,dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    def initialize(self):
        pass

def weight_init(module):
    for n, m in module.named_children():
      #  print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.GELU,nn.ReLU,nn.AdaptiveAvgPool2d,nn.Softmax,nn.MaxPool2d)):
            pass
        else:
            m.initialize()


class ResNet(nn.Module):

    def __init__(self, block, layers,norm_layer=None,base=48):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Sequential(
                    nn.Conv2d(12, base, kernel_size=2, stride=2),
                    LayerNorm(base, eps=1e-6, data_format="channels_first")
            )


        self.layer1_16 = self._make_layer(block, base,base, layers[0],stride=1,dilate=False) #2x
        self.layer2_16 = self._make_layer(block, base,base*2, layers[1], stride=2,dilate=False) #4x
        self.layer3_16 = self._make_layer(block, base*2,base*4, layers[2], stride=2,dilate=False) #8x
        self.layer4_16 = self._make_layer(block, base*4,base*8, layers[3], stride=2,dilate=False) #16x

        # self.layer1_8 = self._make_layer(block, 48,48, layers[4],stride=1,dilate=False)
        self.layer1_8 = self._make_layer(block, base*2,base*2, layers[4],stride=1,dilate=False) #4x
        self.layer2_8 = self._make_layer(block, base*2,base*4, layers[5],stride=2,dilate=False) #8x


        self.layer1_4 = self._make_layer(block, base*2,base*2, layers[6],stride=1,dilate=False) #8x

    def _make_layer(self, block, inplanes,outplanes, blocks, stride=1, dilate=False):
        layers = []
        if stride != 1 :
            downsample = nn.Sequential(
                    LayerNorm(inplanes, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(inplanes, outplanes, kernel_size=2, stride=2),
            )
            layers.append(downsample)

        # layers.append(block(outplanes,outplanes))
        for _ in range(0, blocks):
            layers.append(block(outplanes,outplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out1= self.conv1(x)

        out2_16 = self.layer1_16(out1)
        out3_16 = self.layer2_16(out2_16)
        out4_16 = self.layer3_16(out3_16)
        out5_16 = self.layer4_16(out4_16)

        out4_8  = self.layer1_8(out3_16)
        out5_8  = self.layer2_8(out4_8)

        out5_4  = self.layer1_4(out4_8)


        return { 'out1_2':out2_16,'out1_4':out3_16,'out1_8':out4_16,'out1_16':out5_16,
                 'out2_4':out4_8,'out2_8':out5_8,
                 'out3_4':out5_4}

    def initialize(self):
        weight_init(self)


def _resnet( block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def RFAE(pretrained=False, progress=True, **kwargs):
    return _resnet( BasicBlock, [2, 2, 2, 2, 2, 2,2], pretrained, progress,
                   **kwargs)


