"""
Modified from torchvision, adapted to take 1*28*28 or 3*28*28 Image as input.
"""

import torch
import torch.nn as nn

model_layers = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation,
                     padding=padding, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, **kwargs):
        super(BasicBlock, self).__init__()
        # if dilation == 1:
        #     self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        # elif dilation == 2:
        #     self.conv1 = conv3x3(inplanes, planes, stride, dilation, padding=2)
        # else:
        #     raise ValueError('dilation must be 1 or 2!')
        self.conv1 = conv3x3(inplanes, planes, stride, dilation, padding=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=None, expose_stages=None, dilations=None, stride_in_1x1=False):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        layers_planes = [32, 64, 128, 256]
        layers_strides = [1, 2, 2, 2]
        layers_dilations = dilations if dilations is not None else [1, 1, 1, 1]
        for i, dilation in enumerate(layers_dilations):
            if dilation == 2:
                layers_strides[i] = 1

        for i, (planes, block_num, stride, dilation) in enumerate(zip(layers_planes, layers, layers_strides, layers_dilations)):
            layer = self._make_layer(block, planes, block_num, stride=stride, dilation=dilation, stride_in_1x1=stride_in_1x1)
            self.__setattr__('layer{}'.format(i + 1), layer)

        self.num_layers = i + 1
        self.has_fc_head = True
        if self.has_fc_head:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, num_classes)

        # params_init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, block_num, stride=1, dilation=1, stride_in_1x1=False):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation, stride_in_1x1=stride_in_1x1))
        self.inplanes = planes
        for i in range(1, block_num):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        expose_feats = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for i in range(self.num_layers):
            x = self.__getattr__("layer{}".format(i + 1))(x)
        if self.has_fc_head:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x

    def frozen_parameters(self, frozen_stages=None, frozen_bn=False):
        if frozen_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = False
        if frozen_stages is not None:
            for stage in frozen_stages:
                assert (stage >= 1) and (stage <= 6)
                if stage == 1:
                    for param in self.conv1.parameters():
                        param.requires_grad = False
                    for param in self.bn1.parameters():
                        param.requires_grad = False
                elif stage < 6:
                    for param in self.__getattr__("layer{}".format(stage - 1)).parameters():
                        param.requires_grad = False
                else:
                    for param in self.fc.parameters():
                        param.requires_grad = False

    def bn_eval(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def ResNet18(num_classes=None, expose_stages=None, dilations=None, **kwargs):
    """Constructs a ResNet-18 model
    Args:
        pretrained (bool): if True, load pretrained model. Default: False
        pretrained_model_path (str, optional): only effective when pretrained=True,
                                            if not specified, use pretrained model from model_zoo.
        num_classes (int): number of classes for the fc output score.
        expose_stages (list, optional): list of expose stages, e.g. [4, 5] means expose conv4 and conv5 stage output.
                                        if not specified, only expose output of end_stage.
    """

    if num_classes is None:
        assert expose_stages is not None, "num_class and expose_stages is both None"
        assert 6 not in expose_stages, "can't expose the 6th stage for num_classes is None"

    if expose_stages is None:
        expose_stages = [6]

    end_stage = max(expose_stages)
    assert end_stage <= 6, "the max expose_stage is out of range"

    layers = model_layers['resnet18'][:end_stage - 1]

    model = ResNet(block=BasicBlock, layers=layers, num_classes=num_classes, expose_stages=expose_stages, dilations=dilations)

    return model


def ResNet34(num_classes=None, expose_stages=None, dilations=None, **kwargs):
    """Constructs a ResNet-34 model
    Args:
        pretrained (bool): if True, load pretrained model. Default: False
        pretrained_model_path (str, optional): only effective when pretrained=True,
                                            if not specified, use pretrained model from model_zoo.
        num_classes (int): number of classes for the fc output score.
        expose_stages (list, optional): list of expose stages, e.g. [4, 5] means expose conv4 and conv5 stage output.
                                        if not specified, only expose output of end_stage.
    """

    if num_classes is None:
        assert expose_stages is not None, "num_class and expose_stages is both None"
        assert 6 not in expose_stages, "can't expose the 6th stage for num_classes is None"

    if expose_stages is None:
        expose_stages = [6]

    end_stage = max(expose_stages)
    assert end_stage <= 6, "the max expose_stage is out of range"

    layers = model_layers['resnet34'][:end_stage - 1]

    model = ResNet(block=BasicBlock, layers=layers, num_classes=num_classes, expose_stages=expose_stages,
                   dilations=dilations)

    return model