
import torch
import torch.nn as nn

class LSTM_AE(nn.Module):

  def __init__(self, input_dim, hidden_dim, num_layers):
    super().__init__()

    self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
    self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, proj_size=input_dim)

  def forward(self, x):
    """
    x : input tensor of dim (seq_len, batch_size, input_dim)
    """
    out, _ = self.encoder(x)
    out = out[-1, :, :].unsqueeze(0).repeat(x.shape[0], 1, 1)
    pred_x, _ = self.decoder(out)
    return pred_x
