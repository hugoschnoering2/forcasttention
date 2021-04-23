
import torch
import torch.nn as nn

from models.rnn.lstm_ln import LSTM as LSTM_LN

class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers, layer_norm, dropout):
        super().__init__()
        lstm_module = LSTM_LN if layer_norm else nn.LSTM
        self.lstm = lstm_module(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=embed_dim) if hidden_dim != embed_dim else None
    def forward(self, input):
        x, (hn, cn) = self.lstm(input)
        if self.linear is not None:
            x = self.linear(x)
        return x, (hn, cn)

class LSTMAutoEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, embed_dim, num_layers_encoder, num_layers_decoder, layer_norm=False, dropout=0.1):
    super().__init__()
    self.encoder = LSTMBlock(input_dim, hidden_dim, embed_dim, num_layers_encoder, layer_norm, dropout)
    self.decoder = LSTMBlock(embed_dim, hidden_dim, input_dim, num_layers_decoder, layer_norm, dropout)
  def forward(self, input):
    x, _ = self.encoder(input)
    x = x[-1, :, :].unsqueeze(0).repeat(input.shape[0], 1, 1)
    x, _ = self.decoder(x)
    return x

class LSTMAutoEncoderAll(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers_encoder, num_layers_decoder, layer_norm=False, dropout=0.1):
      super().__init__()
      assert embed_dim < input_dim
      self.encoder = LSTMEncoder(input_dim, hidden_dim, embed_dim, num_layers_encoder, layer_norm, dropout)
      self.decoder = LSTMDecoder(input_dim, hidden_dim, embed_dim, num_layers_decoder, layer_norm, dropout)
    def forward(self, input):
      x, _ = self.encoder(x)
      x, _ = self.decoder(x)
      return x
