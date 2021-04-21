
import torch
import torch.nn as nn

from models.tcn.tcn import TemporalConvNet

class TCN_AE(nn.Module):

  def __init__(self, num_channels, depth, kernel_size, btk_channels, down_factor):
    super().__init__()

    self.tcn_encoder = TemporalConvNet(1, depth*[num_channels], kernel_size)
    self.bottleneck_encoder = nn.Conv1d(num_channels, btk_channels, 1)
    self.downsampling = nn.AvgPool1d(down_factor)

    self.upsampling = nn.Upsample(scale_factor=down_factor, mode='nearest')
    self.tcn_decoder = TemporalConvNet(btk_channels, depth * [num_channels], kernel_size)
    self.bottleneck_decoder = nn.Conv1d(num_channels, 1, 1)

  def forward(self, x):

    h = self.tcn_encoder(x)
    h = self.bottleneck_encoder(h)
    e = self.downsampling(h)

    h = self.upsampling(e)
    h = self.tcn_decoder(h)
    pred_x = self.bottleneck_decoder(h)

    return pred_x
