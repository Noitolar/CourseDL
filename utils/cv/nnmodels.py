import torch
import torch.nn as nn
import torchvision.models as vmodels


class SimpleLinearClassifier(nn.Module):
    def __init__(self, num_classes, input_size=784):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(input_size, 2048), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(2048, 300), nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, inputs: torch.Tensor):
        outputs = self.flatten(inputs)
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
    def __init__(self, num_classes, num_channels=3, from_pretrained=None):
        super().__init__()
        if from_pretrained == "default":
            self.resnet = vmodels.resnet18(weights=vmodels.ResNet18_Weights.DEFAULT)
        elif from_pretrained == "imagenet":
            self.resnet = vmodels.resnet18(weights=vmodels.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = vmodels.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, inputs: torch.Tensor):
        outputs = self.resnet(inputs)
        return outputs


class LprNet(nn.Module):
    def __init__(self, num_classes=66, dropout=0.5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            self.build_basic_block(in_channels=64, out_channels=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            self.build_basic_block(in_channels=64, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            self.build_basic_block(in_channels=256, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=(13, 1)),
            nn.BatchNorm2d(num_features=num_classes),
            nn.ReLU())
        self.extractor = nn.Sequential(nn.Linear(num_classes * 8, 128), nn.ReLU())
        self.container = nn.Conv2d(in_channels=num_classes + 128, out_channels=num_classes, kernel_size=(1, 1))

    @staticmethod
    def build_basic_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=(1, 1))
        )

    def forward(self, inputs):
        inputs = self.backbone(inputs)
        pattern = inputs.flatten(1, -1)
        pattern = self.extractor(pattern)
        pattern = torch.reshape(pattern, [-1, 128, 1, 1])
        pattern = pattern.repeat(1, 1, 1, inputs.size()[-1])
        inputs = torch.cat([inputs, pattern], dim=1)
        inputs = self.container(inputs)
        logits = inputs.squeeze(2)
        return logits
