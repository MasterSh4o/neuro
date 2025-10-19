import torch, torch.nn as nn, torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c//r, 1)
        self.fc2 = nn.Conv2d(c//r, c, 1)
    def forward(self, x):
        s = x.mean(dim=[2,3], keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.se = SEBlock(c_out)
        self.short = nn.Identity() if (c_in == c_out and stride==1) else nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.relu(h + self.short(x))

class InterferoNetMultiLabel(nn.Module):
    def __init__(self, out_dim=50, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock(32, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 384, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x
