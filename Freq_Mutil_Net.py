import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Unet import UNet


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, dilation=1):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=dilation, dilation=dilation)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(in_channels, stride * in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out


class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out


class ERDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(ERDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

        self.calayer = CALayer(in_channels)




    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)

        out = self.calayer(out)
        out = out + x

        return out


class FAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(FAM, self).__init__()
        self.conv1 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv2 = conv(3, n_feat, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        self.conv4 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv5 = conv(n_feat, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_freq):
        x1 = self.conv1(x) + x_freq
        x1 = self.conv3(x1)
        x2 = self.conv2(x_freq)
        x3 = self.conv4(x1 + x2)
        x3 = torch.sigmoid(x3)
        x4 = self.conv5(x1)
        x5 = x3 * x4 + x
        return x5


class HF_Net(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=7, num_dense_layer=4,
                 growth_rate=16, attention=True, freq_inchannels=3, refine_width=5):
        super(HF_Net, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(
            torch.Tensor(np.ones((height + 1, width, 2, depth_rate * stride ** (height - 1)))),
            requires_grad=attention)
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(freq_inchannels + in_channels, in_channels, kernel_size=kernel_size,
                                  padding=(kernel_size - 1) // 2)
        self.rdb_in = ERDB(depth_rate, num_dense_layer, growth_rate)
        # self.rdb_out = RDB(freq_inchannels + in_channels, num_dense_layer, growth_rate)

        # ----refine_block----#
        self.refine_block = UNet(3, 3)
        # self.refine_block = RefineBlock()

        self.conv_mid = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): ERDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        # frequency branch
        self.freq_conv_in = nn.Conv2d(freq_inchannels, depth_rate, kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)
        self.freq_conv_out = nn.Conv2d(depth_rate, freq_inchannels, kernel_size=kernel_size,
                                       padding=(kernel_size - 1) // 2)

        for j in range(width - 1):
            self.rdb_module.update({'{}_{}'.format(3, j): ERDB(depth_rate, num_dense_layer, growth_rate)})

        # FAM module
        self.fam_module = FAM(depth_rate, kernel_size=1, bias=False)

        _in_channels = depth_rate
        for i in range(height - 1):
            for j in range(width // 2):
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):
            for j in range(width // 2, width):
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride

    def forward(self, x, x_freq):
        inp = self.conv_in(x)
        freq_x_index = [[0 for _ in range(self.width)] for _ in range(1)]
        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)

        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j - 1)](x_index[0][j - 1])

        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i - 1, 0)](x_index[i - 1][0])

        for i in range(1, self.height):
            for j in range(1, self.width // 2):
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module[
                    '{}_{}'.format(i, j - 1)](x_index[i][j - 1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module[
                                    '{}_{}'.format(i - 1, j)](x_index[i - 1][j])

        x_index[i][j + 1] = self.rdb_module['{}_{}'.format(i, j)](x_index[i][j])
        k = j

        for j in range(self.width // 2 + 1, self.width):
            x_index[i][j] = self.rdb_module['{}_{}'.format(i, j - 1)](x_index[i][j - 1])

        for i in range(self.height - 2, -1, -1):
            channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
            x_index[i][k + 1] = self.coefficient[i, k + 1, 0, :channel_num][None, :, None, None] * self.rdb_module[
                '{}_{}'.format(i, k)](x_index[i][k]) + \
                                self.coefficient[i, k + 1, 1, :channel_num][None, :, None, None] * self.upsample_module[
                                    '{}_{}'.format(i, k + 1)](x_index[i + 1][k + 1], x_index[i][k].size())

        for i in range(self.height - 2, -1, -1):
            for j in range(self.width // 2 + 1, self.width):
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module[
                    '{}_{}'.format(i, j - 1)](x_index[i][j - 1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module[
                                    '{}_{}'.format(i, j)](x_index[i + 1][j], x_index[i][j - 1].size())

        # freq branch
        freq_in = self.freq_conv_in(x_freq)
        freq_x_index[0][0] = freq_in
        freq_channels_num = self.depth_rate
        for f in range(1, self.width - 1):
            freq_x_index[0][f] = self.coefficient[3, f, 0, :freq_channels_num][None, :, None, None] * self.rdb_module[
                '{}_{}'.format(3, f - 1)](freq_x_index[0][f - 1])  # \

            freq_x_index[0][f] += self.coefficient[3, f, 1, :freq_channels_num][None, :, None, None] * x_index[0][f]

        # FAM
        freq_fam = self.fam_module(freq_x_index[0][-2], x_freq)
        out_freq = F.relu(self.freq_conv_out(freq_fam))
        out_x = F.relu(self.conv_mid(x_index[i][j]))
        out_cat = torch.cat((out_freq, out_x), 1)

        out = F.relu(self.conv_out(out_cat))

        # ----refine block----#
        out = self.refine_block(out)
        return out, out_freq, out_x
