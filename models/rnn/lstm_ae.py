
import torch
import torch.nn as nn

from models.rnn.lstm_ln import LSTM as LSTM_LN

class LSTM_AE(nn.Module):

  def __init__(self, input_dim, hidden_dim, num_layers_encoder, num_layers_decoder, layer_norm=False):
    super().__init__()

    lstm_module = LSTM_LN if layer_norm else nn.LSTM

    self.encoder = lstm_module(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers_encoder)
    self.decoder = lstm_module(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers_decoder, proj_size=input_dim)

  def forward(self, x):
    """
    x : input tensor of dim (seq_len, batch_size, input_dim)
    """
    out, _ = self.encoder(x)
    out = out[-1, :, :].unsqueeze(0).repeat(x.shape[0], 1, 1)
    pred_x, _ = self.decoder(out)
    return pred_x


class LSTM_AE_all(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers_encoder, num_layers_decoder, layer_norm=False):
      super().__init__()
      lstm_module = LSTM_LN if layer_norm else nn.LSTM
      assert hidden_dim < input_dim, "embedding dim is greater than the input dim..."
      self.encoder = lstm_module(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers_encoder)
      self.decoder = lstm_module(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers_decoder)
      self.activation = nn.ReLU()
      self.linear = nn.Linear(in_features=hidden_dim, out_features=input_dim)

    def forward(self, x):
      """
      x : input tensor of dim (seq_len, batch_size, input_dim)
      """
      out, _ = self.encoder(x)
      out, _ = self.decoder(out)
      pred_x = self.linear(self.activation(out))
      return pred_x
