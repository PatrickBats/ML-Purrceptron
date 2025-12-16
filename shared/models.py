import torch
import torch.nn as nn
import torchvision.models as models


class CatCNN(nn.Module):

    def __init__(self, num_classes=8, dropout_rate=0.5):
        super(CatCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate flattened size: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        # 512 channels * 7 * 7 = 25088
        self.flatten_size = 512 * 7 * 7

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6)  # Less dropout in second FC layer
        )

        self.fc3 = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNetTransfer(nn.Module):

    def __init__(self, num_classes=8, freeze_backbone=False):
        super(ResNetTransfer, self).__init__()

        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)

        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace final FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_type='from_scratch', num_classes=8, **kwargs):
    if model_type == 'from_scratch':
        return CatCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'transfer_learning':
        return ResNetTransfer(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    model_scratch = CatCNN(num_classes=8)
    model_transfer = ResNetTransfer(num_classes=8)

    print(f"From-scratch params: {model_scratch.get_num_params():,}")
    print(f"Transfer params: {model_transfer.get_num_params():,}")
