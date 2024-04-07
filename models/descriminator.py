# Simple Discriminator
import kornia
import torch.nn as nn
import torch.nn.functional as F

class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.maxblurpool = kornia.filters.MaxBlurPool2D(kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.maxblurpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.maxblurpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)
        x = self.maxblurpool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu(x)
        x = self.conv5(x)

        return x