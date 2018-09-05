import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from tcn import TemporalConvNet


class encoder(nn.Module):
    def __init__(self, num_inputs,num_outputs ,num_channels, kernel_size=2, dropout=0.2):
        super(encoder, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.conv1d = nn.ConvTranspose1d(num_channels[-1], num_outputs, kernel_size=6, stride=2)
        self.linear = nn.Linear(num_outputs, num_outputs)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 1)

    def forward(self, x):
        y = self.tcn(x)
        y = self.conv1d(y)
        return self.linear(y.transpose(1, 2))

class decoder(nn.Module):
    def __init__(self, num_inputs,num_outputs ,num_channels, kernel_size=2, dropout=0.2):
        super(decoder, self).__init__()
        self.upper = nn.Upsample(scale_factor=12.5, mode='bilinear', align_corners=True)
        self.tcn = TemporalConvNet(num_inputs=int(237), num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], num_outputs)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 1)

    def forward(self, x):
        y = x.view(1, -1, 10, 19)
        y = self.upper(y)
        y = y.view(-1, 125, 237)
        y = y.transpose(1, 2)
        y = self.tcn(y)  
        y = y.transpose(1, 2)
        y = self.linear(y)
        return y