import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from einops import rearrange
from new_conv.wtconv2d import WTConv2d
from XAI import layers_mm24 as lym
from XAI.layers_mm24 import ExplainFrame

class InceptionConv2d(nn.Module, ExplainFrame):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        k1 = 255
        k2 = 127
        k3 = 63
        self.conv1 = lym.Conv2d(in_channels, out_channels, kernel_size=(1,k1), stride=1, padding=(0,k1 // 2), bias=bias)
        self.conv2 = lym.Conv2d(in_channels, out_channels, kernel_size=(1,k2), stride=1, padding=(0,k2 // 2), bias=bias)
        self.conv3 = lym.Conv2d(in_channels, out_channels, kernel_size=(1,k3), stride=1, padding=(0,k3 // 2), bias=bias)
        self.v1 = None
        self.v2 = None
        self.v3 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v1 = self.conv1(x)
        self.v2 = self.conv2(x)
        self.v3 = self.conv3(x)
        return self.v1 + self.v2 + self.v3


    def explain(self, R):
        R1 = R * self.v1 / self.Y
        R2 = R * self.v2 / self.Y
        R3 = R * self.v3 / self.Y

        R1 = self.conv1.explain(R1)
        R2 = self.conv2.explain(R2)
        R3 = self.conv3.explain(R3)

        return R1 + R2 + R3



class P4(nn.Module):
    def __init__(
            self,
            F1=8,
            F2=16,
            D=2,
            nt=512,
            nc=96,
            classes=40,
            dropout=0.5,
    ):
        super().__init__()

        K1 = nt // 2

        self.conv0 = InceptionConv2d(1, F1)####################只改通道
        self.bn0 = lym.BatchNorm2d(F1)

        # self.conv1 = nn.Sequential(
        #     nn.ZeroPad2d((K1 // 2 - 1, K1 // 2, 0, 0)),
        #     nn.Conv2d(1, F1, (1, K1), bias=False, stride=(1, 1)),
        # )
        # self.bn1 = nn.BatchNorm2d(F1)

        self.re1 = Rearrange("b f c t -> b c f t")###############
        self.adctconv = WTConv2d(96)   #  WTConv2d; ADCTConv2d
        self.re2 = Rearrange("b c f t -> b f c t")

        self.conv2 = lym.Conv2d(F1, F1 * D, (nc, 1), bias=False, groups=F1)
        self.bn2 = lym.BatchNorm2d(F1 * D)

        # self.act1 = nn.ELU() ############################################
        # self.act1 = nn.Softshrink()
        self.pool1 = lym.AdaptiveAvgPool2d((1, 4))
        self.dropout1 = lym.Dropout(dropout)


        # 时间卷积-1x1卷积------>1x1卷积
        self.conv4 = lym.Conv2d(F1 * D, F2, (1, 1), bias=False)

        self.bn3 = lym.BatchNorm2d(F2)
        self.act = lym.ELU()
        self.pool2 = lym.AdaptiveAvgPool2d((1, 8))
        self.dropout2 = lym.Dropout(dropout)

        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(F2 * (nt // 32), classes)
        self.fc = lym.Linear(F2 * (nt // 32), classes)

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.conv0(x)
        x = self.bn0(x)
        print("After conv0 shape:", x.shape)

        x = self.re1(x)
        x = self.adctconv(x)
        x = self.re2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x, None


    def explain(self, y, class_index, class_num):
        if class_index is None:
            R = torch.eye(class_num)[torch.max(y, 1)[1]].to(y.device)
        else:
            R = torch.eye(class_num)[class_index].to(y.device)
        R = torch.clamp(R*y, min=0)
        # R = R*y

        R = self.fc.explain(R)
        R = rearrange(R, 'b (a b c) -> b a b c ', a=16, b=1, c=16)

        R = self.dropout2.explain(R)
        R = self.pool2.explain(R)
        R= self.act.explain(R)
        R = self.bn3.explain(R)
        R = self.conv4.explain(R)

        R = self.dropout1.explain(R)
        R = self.pool1.explain(R)

        R = self.act.explain(R)
        R = self.bn2.explain(R)
        R = self.conv2.explain(R)

        R = rearrange(R, "b f c t -> b c f t")
        R = self.adctconv.explain(R)
        R = rearrange(R, 'b c f t -> b f c t')  # rearrange

        R = self.bn0.explain(R)
        R = self.conv0.explain(R)  # InceptionConv2d
        return R

