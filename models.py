import torch.nn as nn


class Var1(nn.Module):
    """
    base Var
    """
    def __init__(self, num_classes, in_channels=3):
        super(Var1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),  # 160 * 120 -> 158 * 118
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3),  # 158 * 118 -> 156 * 116
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 156 * 116 -> 78 * 58

        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3),  # 78 * 58 -> 76 * 56
            nn.BatchNorm2d(48),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3),  # 76 * 56 -> 74 * 54
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 74 * 54 -> 37 * 27

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3),  # 37 * 27 -> 35 * 25
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 35 * 25 -> 17 * 12

        self.fc = nn.Sequential(
            nn.Linear(96 * 17 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Var2(nn.Module):
    """
    wider and deeper Var
    """
    def __init__(self, num_classes, in_channels=3):
        super(Var2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),  # 160 * 120 -> 158 * 118
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # 158 * 118 -> 156 * 116
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 156 * 116 -> 78 * 58

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),  # 78 * 58 -> 76 * 56
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(.2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 76 * 56 -> 74 * 54
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 74 * 54 -> 37 * 27

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 37 * 27
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(.2))

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 37 * 27
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 37 * 27 -> 18 * 13

        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),  # 18 * 13 -> 16 * 11
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(.4))  # 16 * 11 -> 8 * 5

        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Var3(nn.Module):
    """
    first Var with 4 pooling layers
    """
    def __init__(self, num_classes, in_channels=3):
        super(Var3, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3),  # 160 * 120 -> 158 * 118
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 78 * 59

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),  # 78 * 59 -> 76 * 57
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 76 * 57 -> 38 * 28

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # 38 * 28 -> 36 * 26
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 36 * 26 -> 18 * 13

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 18 * 13
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 18 * 13 -> 9 * 6

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(96 * 9 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Var4(nn.Module):
    """
    architecture with dropout at 4 layers
    """
    def __init__(self, num_classes, in_channels=3):
        super(Var4, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),  # 160 * 120 -> 158 * 118
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # 158 * 118 -> 156 * 116
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 156 * 116 -> 78 * 58

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),  # 78 * 58 -> 76 * 56
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(.2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 76 * 56 -> 74 * 54
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(.2))  # 74 * 54 -> 37 * 27

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),  # 37 * 27
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(.2))

        self.layer6 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),  # 37 * 27
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(.25))  # 37 * 27 -> 18 * 13

        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),  # 18 * 13 -> 16 * 11
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(.2))  # 16 * 11 -> 8 * 5

        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Var5(nn.Module):
    """
    smaller Var
    """
    def __init__(self, num_classes, in_channels=3):
        super(Var5, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3),  # 160 * 120 -> 158 * 118
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),  # 158 * 118 -> 156 * 116
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 156 * 116 -> 78 * 58

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # 78 * 58 -> 76 * 56
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 78 * 58 -> 38 * 28

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),  # 38 * 28 -> 36 * 26
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 36 * 26 -> 18 * 13

        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 18 * 13 -> 16 * 11
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 16 * 11 -> 8 * 5

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Var6(nn.Module):
    """
    same as Var 4 but with Mish activation instead of ReLu
    """
    def __init__(self, num_classes, in_channels=3):
        super(Var6, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),  # 160 * 120 -> 158 * 118
            nn.BatchNorm2d(16),
            nn.Mish())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # 158 * 118 -> 156 * 116
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 156 * 116 -> 78 * 58

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),  # 78 * 58 -> 76 * 56
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Dropout(.2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 76 * 56 -> 74 * 54
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(.2))  # 74 * 54 -> 37 * 27

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),  # 37 * 27
            nn.BatchNorm2d(96),
            nn.Mish(),
            nn.Dropout(.2))

        self.layer6 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),  # 37 * 27
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(.25))  # 37 * 27 -> 18 * 13

        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),  # 18 * 13 -> 16 * 11
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(.2))  # 16 * 11 -> 8 * 5

        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 5, 128),
            nn.Mish(),
            nn.Linear(128, 32),
            nn.Mish(),
            nn.Linear(32, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Var7(nn.Module):
    """
    fully convolutional variation with average max pooling and 5 Maxpool
    """
    def __init__(self, num_classes, in_channels=3):
        super(Var7, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1, padding_mode='circular'),  # 160 * 120
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 160 * 120 -> 80 * 60

        self.layer3 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 80 * 60 -> 40 * 30
            nn.Dropout(.2))

        self.layer4 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 40 * 30 -> 20 * 15
            nn.Dropout(.2))

        self.layer5 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 20 * 15 -> 10 * 7
            nn.Dropout(.2))

        self.layer6 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10 * 7 -> 5 * 3
            nn.Dropout(.25))

        self.layer7 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(.2))

        self.fc = nn.Sequential(
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.max(2)[0].max(2)[0]
        x = self.fc(x)
        return x


