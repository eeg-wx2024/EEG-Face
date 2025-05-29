import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct


class DCTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True):
        super(DCTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.stride = stride
        self.dilation = 1

        # 基本卷积层，作用在输入图像上
        self.base_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            stride=1,
            dilation=1,
            groups=in_channels,
            bias=bias,
        )
        # 缩放模块，用于调整卷积后的张量
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        if self.stride > 1:
            self.stride_filter = nn.Parameter(
                torch.ones(in_channels, 1, 1, 1), requires_grad=False
            )
            self.do_stride = lambda x_in: F.conv2d(
                x_in,
                self.stride_filter,
                bias=None,
                stride=self.stride,
                groups=in_channels,
            )
        else:
            self.do_stride = None

    def forward(self, x):
        # 计算DCT
        x_dct = dct.dct_2d(x)
        x_dct = dct.dct_2d(x_dct)

        # 卷积操作
        x_conv = self.base_scale(self.base_conv(x_dct))

        # 逆DCT
        x_idct = dct.dct_2d(x_conv)
        x_idct = dct.idct_2d(x_idct)

        if self.do_stride is not None:
            x_idct = self.do_stride(x_idct)

        return x_idct


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


if __name__ == "__main__":
    x = torch.zeros(2, 3, 10, 10)
    dct_conv = DCTConv2d(3, 3, kernel_size=5, stride=2)

    y = dct_conv(x)
    print(y.shape)
