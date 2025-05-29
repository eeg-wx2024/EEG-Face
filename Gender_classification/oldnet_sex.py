import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
import datetime
import os
from torch import Tensor
from einops.layers.torch import Rearrange
from sklearn.neighbors import KNeighborsClassifier


class classifier_EEGNet(nn.Module):
    def __init__(self, spatial=126, temporal=500):
        super(classifier_EEGNet, self).__init__()
        # possible spatial [128, 96, 64, 32, 16, 8]
        # possible temporal [1024, 512, 440, 256, 200, 128, 100, 50]
        F1 = 8
        F2 = 16
        D = 2
        first_kernel = 64
        first_padding = first_kernel // 2
        self.network = nn.Sequential(
            nn.ZeroPad2d((first_padding, first_padding - 1, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, first_kernel)),
            nn.BatchNorm2d(F1),
            nn.Conv2d(
                in_channels=F1, out_channels=F1, kernel_size=(spatial, 1), groups=F1
            ),
            nn.Conv2d(in_channels=F1, out_channels=D * F1, kernel_size=1),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(),
            nn.ZeroPad2d((8, 7, 0, 0)),
            nn.Conv2d(
                in_channels=D * F1, out_channels=D * F1, kernel_size=(1, 16), groups=F1
            ),
            nn.Conv2d(in_channels=D * F1, out_channels=F2, kernel_size=1),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(),
        )
        self.fc = nn.Linear(F2 * (temporal // 32), 40)

    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size()[0], -1)
        return self.fc(x)


##############################################################
# SyncNet classifier
##############################################################


class classifier_SyncNet(nn.Module):
    def __init__(self, spatial=126, temporal=500):
        super(classifier_SyncNet, self).__init__()
        K = min(10, spatial)
        Nt = min(40, temporal)
        pool_size = Nt
        b = np.random.uniform(low=-0.05, high=0.05, size=(1, spatial, K))
        omega = np.random.uniform(low=0, high=1, size=(1, 1, K))
        zeros = np.zeros(shape=(1, 1, K))
        phi_ini = np.random.normal(loc=0, scale=0.05, size=(1, spatial - 1, K))
        phi = np.concatenate([zeros, phi_ini], axis=1)
        beta = np.random.uniform(low=0, high=0.05, size=(1, 1, K))
        t = np.reshape(range(-Nt // 2, Nt // 2), [Nt, 1, 1])
        tc = np.single(t)
        W_osc = b * np.cos(tc * omega + phi)
        W_decay = np.exp(-np.power(tc, 2) * beta)
        W = W_osc * W_decay
        W = np.transpose(W, (2, 1, 0))
        bias = np.zeros(shape=[K])
        self.net = nn.Sequential(
            nn.ConstantPad1d((Nt // 2, Nt // 2 - 1), 0),
            nn.Conv1d(
                in_channels=spatial, out_channels=K, kernel_size=1, stride=1, bias=True
            ),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            nn.ReLU(),
        )
        self.net[1].weight.data = torch.FloatTensor(W)
        self.net[1].bias.data = torch.FloatTensor(bias)
        self.fc = nn.Linear((temporal // pool_size) * K, 40)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.net(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


#############################################################################
# EEG-ChannelNet classifier
#############################################################################


class classifier_EEGChannelNet(nn.Module):

    def __init__(self, spatial=126, temporal=500):
        super(classifier_EEGChannelNet, self).__init__()
        self.temporal_layers = nn.ModuleList([])
        self.temporal_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=10,
                    kernel_size=(1, 33),
                    stride=(1, 2),
                    dilation=(1, 1),
                    padding=(0, 16),
                ),
                nn.BatchNorm2d(10),
                nn.ReLU(),
            )
        )
        self.temporal_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=10,
                    kernel_size=(1, 33),
                    stride=(1, 2),
                    dilation=(1, 2),
                    padding=(0, 32),
                ),
                nn.BatchNorm2d(10),
                nn.ReLU(),
            )
        )
        self.temporal_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=10,
                    kernel_size=(1, 33),
                    stride=(1, 2),
                    dilation=(1, 4),
                    padding=(0, 64),
                ),
                nn.BatchNorm2d(10),
                nn.ReLU(),
            )
        )
        self.temporal_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=10,
                    kernel_size=(1, 33),
                    stride=(1, 2),
                    dilation=(1, 8),
                    padding=(0, 128),
                ),
                nn.BatchNorm2d(10),
                nn.ReLU(),
            )
        )
        self.temporal_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=10,
                    kernel_size=(1, 33),
                    stride=(1, 2),
                    dilation=(1, 16),
                    padding=(0, 256),
                ),
                nn.BatchNorm2d(10),
                nn.ReLU(),
            )
        )
        self.spatial_layers = nn.ModuleList([])
        self.spatial_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=50,
                    out_channels=50,
                    kernel_size=(128, 1),
                    stride=(2, 1),
                    padding=(63, 0),
                ),
                nn.BatchNorm2d(50),
                nn.ReLU(),
            )
        )
        self.spatial_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=50,
                    out_channels=50,
                    kernel_size=(64, 1),
                    stride=(2, 1),
                    padding=(31, 0),
                ),
                nn.BatchNorm2d(50),
                nn.ReLU(),
            )
        )
        self.spatial_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=50,
                    out_channels=50,
                    kernel_size=(32, 1),
                    stride=(2, 1),
                    padding=(15, 0),
                ),
                nn.BatchNorm2d(50),
                nn.ReLU(),
            )
        )
        self.spatial_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=50,
                    out_channels=50,
                    kernel_size=(16, 1),
                    stride=(2, 1),
                    padding=(7, 0),
                ),
                nn.BatchNorm2d(50),
                nn.ReLU(),
            )
        )
        self.residual_layers = nn.ModuleList([])
        self.residual_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=200,
                    out_channels=200,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=200,
                    out_channels=200,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(200),
            )
        )
        self.residual_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=200,
                    out_channels=200,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=200,
                    out_channels=200,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(200),
            )
        )
        self.residual_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=200,
                    out_channels=200,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=200,
                    out_channels=200,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(200),
            )
        )
        self.residual_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=200,
                    out_channels=200,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=200,
                    out_channels=200,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(200),
            )
        )
        self.shortcuts = nn.ModuleList([])
        self.shortcuts.append(
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=1, stride=2),
                nn.BatchNorm2d(200),
            )
        )
        self.shortcuts.append(
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=1, stride=2),
                nn.BatchNorm2d(200),
            )
        )
        self.shortcuts.append(
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=1, stride=2),
                nn.BatchNorm2d(200),
            )
        )
        self.shortcuts.append(
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=1, stride=2),
                nn.BatchNorm2d(200),
            )
        )
        spatial_kernel = 3
        temporal_kernel = 3
        if spatial == 128:
            spatial_kernel = 3
        elif spatial == 96:
            spatial_kernel = 3
        elif spatial == 64:
            spatial_kernel = 2
        else:
            spatial_kernel = 1
        if temporal == 1024:
            temporal_kernel = 3
        elif temporal == 512:
            temporal_kernel = 3
        elif temporal == 440:
            temporal_kernel = 3
        elif temporal == 50:
            temporal_kernel = 2
        self.final_conv = nn.Conv2d(
            in_channels=200,
            out_channels=50,
            kernel_size=(spatial_kernel, temporal_kernel),
            stride=1,
            dilation=1,
            padding=0,
        )
        spatial_sizes = [128, 96, 64, 32, 16, 8, 126]
        spatial_outs = [2, 1, 1, 1, 1, 1, 1]
        temporal_sizes = [1024, 512, 440, 256, 200, 128, 100, 50, 500]
        temporal_outs = [30, 14, 12, 6, 5, 2, 2, 1, 5, 500]
        inp_size = (
            50
            * spatial_outs[spatial_sizes.index(spatial)]
            * temporal_outs[temporal_sizes.index(temporal)]
        )
        self.fc1 = nn.Linear(inp_size, 1000)
        self.fc2 = nn.Linear(1000, 40)

    def forward(self, x):
        y = []
        for i in range(5):
            y.append(self.temporal_layers[i](x))
        x = torch.cat(y, 1)
        y = []
        for i in range(4):
            y.append(self.spatial_layers[i](x))
        x = torch.cat(y, 1)
        for i in range(4):
            x = F.relu(self.shortcuts[i](x) + self.residual_layers[i](x))
        x = self.final_conv(x)
        x = x.view(x.size()[0], -1)  # 展平张量
        # 动态调整 fc1 的输入尺寸
        if self.fc1.in_features != x.size(1):
            self.fc1 = nn.Linear(x.size(1), 1000).to(x.device)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class classifier_MLP(nn.Module):
    def __init__(self, in_channel=126, num_points=500, num_classes=40):
        super(classifier_MLP, self).__init__()

        # Define the layers for the MLP model
        self.fc1 = nn.Linear(in_channel * num_points, 500)  # Flatten the input to (96*512)
        self.fc4 = nn.Linear(500, num_classes)              # Output layer for classification

        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Flatten the input from (b, 1, 96, 512) to (b, 96*512)
        x = x.view(x.size(0), -1)  # x.size(0) is the batch size, and -1 automatically calculates the remaining dimension

        # Pass the input through the network layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after each ReLU layer

        # Output the classification logits
        x = self.fc4(x)
        return x

class classifier_CNN(nn.Module):

    def __init__(self, in_channel=126, num_points=500):
        super(classifier_CNN, self).__init__()
        self.channel = in_channel
        conv1_size = 32
        conv1_stride = 1
        self.conv1_out_channels = 8
        self.conv1_out = int(math.floor(((num_points - conv1_size) / conv1_stride + 1)))
        fc1_in = self.channel * self.conv1_out_channels
        fc1_out = 40
        pool1_size = 128
        pool1_stride = 64
        pool1_out = int(math.floor(((self.conv1_out - pool1_size) / pool1_stride + 1)))
        dropout_p = 0.5
        fc2_in = pool1_out * fc1_out
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.conv1_out_channels,
            kernel_size=conv1_size,
            stride=conv1_stride,
        )
        self.fc1 = nn.Linear(fc1_in, fc1_out)
        self.pool1 = nn.AvgPool1d(kernel_size=pool1_size, stride=pool1_stride)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(fc2_in, 40)

    def forward(self, x):
        batch_size = x.data.shape[0]
        x = x.transpose(1, 2)
        x = x.contiguous().view(-1, 1, x.data.shape[-1])
        x = self.conv1(x)
        x = self.activation(x)
        x = x.view(batch_size, self.channel, self.conv1_out_channels, self.conv1_out)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(
            batch_size, self.conv1_out, self.channel * self.conv1_out_channels
        )
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.pool1(x)
        x = x.contiguous().view(batch_size, -1)
        x = self.fc2(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self, num_electrodes: int, hid_channels: int = 40, dropout: float = 0.5
    ):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, hid_channels, (1, 25), (1, 1)),
            nn.Conv2d(hid_channels, hid_channels, (num_electrodes, 1), (1, 1)),
            nn.BatchNorm2d(hid_channels),
            nn.ELU(),
            nn.AvgPool2d(
                (1, 75), (1, 15)
            ),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(
                hid_channels, hid_channels, (1, 1), stride=(1, 1)
            ),  # transpose, conv could enhance fiting ability slightly
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        # b 1 96 512 -> b 40 1 28
        x = self.shallownet(x)
        # b 40 1 28 -> b 28 40
        x = self.projection(x)
        return x
class MultiHeadAttention(nn.Module):
    def __init__(self, hid_channels: int, heads: int, dropout: float):
        super().__init__()
        self.hid_channels = hid_channels
        self.heads = heads
        self.keys = nn.Linear(hid_channels, hid_channels)
        self.queries = nn.Linear(hid_channels, hid_channels)
        self.values = nn.Linear(hid_channels, hid_channels)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(hid_channels, hid_channels)
        
        # 预定义 Rearrange 层
        self.to_queries = Rearrange("b n (h d) -> b h n d", h=self.heads)
        self.to_keys = Rearrange("b n (h d) -> b h n d", h=self.heads)
        self.to_values = Rearrange("b n (h d) -> b h n d", h=self.heads)
        self.to_output = Rearrange("b h n d -> b n (h d)")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = self.to_queries(self.queries(x))
        keys = self.to_keys(self.keys(x))
        values = self.to_values(self.values(x))
        
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.hid_channels ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = self.to_output(out)
        out = self.projection(out)
        return out
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
class FeedForwardBlock(nn.Sequential):
    def __init__(self, hid_channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__(
            nn.Linear(hid_channels, expansion * hid_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * hid_channels, hid_channels),
        )
class GELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        hid_channels: int,
        heads: int,
        dropout: float,
        forward_expansion: int,
        forward_dropout: float,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(hid_channels),
                    MultiHeadAttention(hid_channels, heads, dropout),
                    nn.Dropout(dropout),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(hid_channels),
                    FeedForwardBlock(
                        hid_channels,
                        expansion=forward_expansion,
                        dropout=forward_dropout,
                    ),
                    nn.Dropout(dropout),
                )
            ),
        )
class TransformerEncoder(nn.Sequential):
    def __init__(
        self,
        depth: int,
        hid_channels: int,
        heads: int = 10,
        dropout: float = 0.5,
        forward_expansion: int = 4,
        forward_dropout: float = 0.5,
    ):
        super().__init__(
            *[
                TransformerEncoderBlock(
                    hid_channels=hid_channels,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    forward_dropout=forward_dropout,
                )
                for _ in range(depth)
            ]
        )
class ClassificationHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hid_channels: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hid_channels * 8),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_channels * 8, hid_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_channels, num_classes),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x

# NICE-EEG
class TSConv(nn.Module):
    def __init__(self, nc=126, nt=500, num_classes=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (nc, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        # 动态计算全连接层的输入维度
        with torch.no_grad():
            mock_input = torch.randn(1, 1, nc, nt)
            mock_output = self.tsconv(mock_input)
            fc_input_dim = mock_output.view(1, -1).shape[1]
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.tsconv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Conformer(nn.Module):
    def __init__(
        self,
        nc: int = 126,
        nt: int = 500,
        embed_dropout: float = 0.5,
        hid_channels: int = 40,
        depth: int = 6,
        heads: int = 10,
        dropout: float = 0.5,
        forward_expansion: int = 4,
        forward_dropout: float = 0.5,
        cls_channels: int = 32,
        cls_dropout: float = 0.5,
        num_classes: int = 40,
    ):
        super().__init__()
        self.num_electrodes = nc
        self.sampling_rate = nt
        self.embed_dropout = embed_dropout
        self.hid_channels = hid_channels
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        self.forward_dropout = forward_dropout
        self.cls_channels = cls_channels
        self.cls_dropout = cls_dropout
        self.num_classes = num_classes

        self.embd = PatchEmbedding(nc, hid_channels, embed_dropout)
        self.encoder = TransformerEncoder(
            depth,
            hid_channels,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
            forward_dropout=forward_dropout,
        )
        self.cls = ClassificationHead(
            in_channels=self.feature_dim(),
            num_classes=num_classes,
            hid_channels=cls_channels,
            dropout=cls_dropout,
        )

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.sampling_rate)

            mock_eeg = self.embd(mock_eeg)
            mock_eeg = self.encoder(mock_eeg)
            mock_eeg = mock_eeg.flatten(start_dim=1)

            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embd(x)
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.cls(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=500, num_layers=2, output_size=40):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层，将 LSTM 的输出映射到最终的输出
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape input if it comes in as (batch, 1, channels, time)
        if x.dim() == 4 and x.shape[1] == 1:
            # x shape: (batch_size, 1, 96, 512) -> (batch_size, 512, 96)
            batch_size, _, channels, time_steps = x.shape
            x = x.squeeze(1) # Remove channel dim (assuming it's 1) -> (batch, 96, 512)
            x = x.permute(0, 2, 1) # Permute to (batch, 512, 96)

        # 初始化 LSTM 的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # 取 LSTM 的最后一个时间步的输出，并送入全连接层
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

class MultiClassSVM(nn.Module):
    def __init__(self, nc=126, nt=500, num_classes=40, lambda_reg=0.01):
        super(MultiClassSVM, self).__init__()
        self.linear = nn.Linear(nc * nt, num_classes)
        self.lambda_reg = lambda_reg

    def forward(self, x):
        x = x.view(x.size(0), -1)

        return self.linear(x)

    def hinge_loss(self, outputs, labels):
        """
        计算多类合页损失
        outputs: Tensor of shape (batch_size, num_classes)
        labels: Tensor of shape (batch_size,)
        """
        # 获取正确类别的分数
        correct_class_scores = outputs[torch.arange(outputs.size(0)), labels].unsqueeze(
            1
        )
        # 计算margin
        margins = F.relu(outputs - correct_class_scores + 1)
        # 正确类别的margin设为0
        margins[torch.arange(outputs.size(0)), labels] = 0
        # 计算平均损失
        loss = margins.sum() / outputs.size(0)
        # 添加正则化项
        loss += self.lambda_reg * torch.sum(self.linear.weight**2)
        return loss

class KNNClassifier(nn.Module):
    def __init__(self, n_neighbors=5):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_fitted = False

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before inference.")
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x_np = x.cpu().numpy().reshape(x.shape[0], -1)
        pred = self.model.predict(x_np)  # 直接返回二分类预测结果（0或1）
        return torch.from_numpy(pred).float().to(x.device).unsqueeze(1)  # 形状: (batch_size, 1)

    def fit(self, X, y):
        # 确保标签是二分类的（0或1）
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = np.asarray(y)
        assert set(np.unique(y_np)).issubset({0, 1}), "Labels must be binary (0 or 1)."

        # 处理输入数据
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X)
        X_np = X_np.reshape(X_np.shape[0], -1)  # 展平

        self.model.fit(X_np, y_np)
        self.is_fitted = True

class OldNet(nn.Module):
    def __init__(self, model_name='oldnet'):
        super(OldNet, self).__init__()
        if model_name == 'eeg':
            self.net = classifier_EEGNet()
            self.net.fc = nn.Linear(self.net.fc.in_features, 1)  # 修改为输出 1 个值
        elif model_name == 'sync':
            self.net = classifier_SyncNet()
            self.net.fc = nn.Linear(self.net.fc.in_features, 1)  # 修改为输出 1 个值
        elif model_name == 'chan':
            self.net = classifier_EEGChannelNet()
            self.net.fc2 = nn.Linear(self.net.fc2.in_features, 1)  # 修改为输出 1 个值
        elif model_name == 'mlp':
            self.net = classifier_MLP()
            self.net.fc4 = nn.Linear(self.net.fc4.in_features, 1)  # 修改为输出 1 个值
        elif model_name == 'cnn':
            self.net = classifier_CNN()
            self.net.fc2 = nn.Linear(self.net.fc2.in_features, 1)  # 修改为输出 1 个值
        elif model_name == 'nice':
            self.net = TSConv()
            self.net.fc = nn.Linear(self.net.fc.in_features, 1)  # 修改为输出 1 个值
        elif model_name == 'svm':
            self.net = MultiClassSVM()
            self.net.linear = nn.Linear(self.net.linear.in_features, 1)  # 修改为输出 1 个值
        elif model_name == 'lstm':
            self.net = LSTMModel()
            self.net.fc = nn.Linear(self.net.fc.in_features, 1)  # 修改为输出 1 个值
        elif model_name == 'con':
            self.net = Conformer()
            self.net.cls = ClassificationHead(
                in_channels=self.net.feature_dim(),
                num_classes=1,  # 修改为输出 1 个值
                hid_channels=self.net.cls_channels,
                dropout=self.net.cls_dropout,
            )
        elif model_name == 'knn':
            self.net = KNNClassifier(n_neighbors=5)
        else:
            self.net = classifier_EEGNet()
            self.net.fc = nn.Linear(self.net.fc.in_features, 1)  # 修改为输出 1 个值

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    device = "cuda:0"
    net = OldNet().to(device)
    x = torch.randn(2, 1, 96, 512).to(device)
    print(net(x).shape)

def print_result(final_accs):
    timestamp = datetime.datetime.now().strftime("%y%m%d")
    model_type = os.path.basename(__file__).split('_')[-1].split('.')[0]
    log_dir = os.path.join(model_type, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "results.log")
    
    with open(log_file, 'w') as f:
        f.write(f"Top-1 Accuracies (5 folds): {final_accs}\n")
        f.write(f"Top-5 Accuracies (5 folds): {final_accs}\n")
        f.write(f"Mean Top-1: {np.mean(final_accs):.2f}\n")
        f.write(f"Mean Top-5: {np.mean(final_accs):.2f}\n")
