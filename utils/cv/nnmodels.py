import torch
import torch.nn as nn
import torchvision.models as vmodels


class SimpleLinearClassifier(nn.Module):
    def __init__(self, num_classes, input_size=784):
        super().__init__()
        self.faltten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(input_size, 2048), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(2048, 300), nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, inputs: torch.Tensor):
        outputs = self.faltten(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs


class SimpleConvClassifier(nn.Module):
    def __init__(self, num_classes, num_channels):
        super().__init__()
        self.layer1 = self.build_layer(num_channels, 16)
        self.layer2 = self.build_layer(16, 32)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    @staticmethod
    def build_layer(conv_in_channels, conv_out_channels, conv_kernel_size=5, conv_stride=1, conv_padding=2, pool_kernel_size=2):
        layer = nn.Sequential(
            nn.Conv2d(conv_in_channels, conv_out_channels, conv_kernel_size, conv_stride, conv_padding),
            nn.ReLU(), nn.BatchNorm2d(conv_out_channels), nn.MaxPool2d(pool_kernel_size))
        return layer

    def forward(self, inputs):
        outputs = self.layer1(inputs)
        outputs = self.layer2(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc(outputs)
        return outputs


class Resnet18Classifier(nn.Module):
    def __init__(self, num_classes, num_channels=3, use_pretrained=False):
        super().__init__()
        self.resnet = vmodels.resnet18(weights=use_pretrained)
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, inputs: torch.Tensor):
        outputs = self.resnet(inputs)
        return outputs
