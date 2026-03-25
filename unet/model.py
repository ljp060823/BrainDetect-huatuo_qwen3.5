import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):  # Depthwise Separable Conv 优化
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class CBAM(nn.Module):  # 简单 Attention（Channel + Spatial）
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ca = self.channel_att(x)
        x = x * ca
        sa = self.spatial_att(x)
        return x * sa

class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):  # 9类（bg+8）
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder (DSConv 省显存)
        self.inc = DSConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DSConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DSConv(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DSConv(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = DSConv(512, 512)
        
        # Decoder + Attention
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att1 = CBAM(768)
        self.conv5 = DSConv(768, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = CBAM(384)
        self.conv6 = DSConv(384, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att3 = CBAM(192)
        self.conv7 = DSConv(192, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.outc = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.conv4(self.down4(x4))
        
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.att1(x)
        x = self.conv5(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.att2(x)
        x = self.conv6(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.att3(x)
        x = self.conv7(x)
        
        x = self.up4(x)
        x = self.outc(x)
        return x  # 返回 logits，训练时用 CrossEntropyLoss