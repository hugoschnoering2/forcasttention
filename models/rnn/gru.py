
import torch
import torch.nn as nn

class GRUBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=embed_dim) if hidden_dim != embed_dim else None
    def forward(self, input):
        x, hn = self.gru(input)
        if self.linear is not None:
            x = self.linear(x)
        return x, hn

class GRUAutoEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, embed_dim, num_layers_encoder, num_layers_decoder, dropout=0.1):
    super().__init__()
    self.encoder = GRUBlock(input_dim, hidden_dim, embed_dim, num_layers_encoder, dropout)
    self.decoder = GRUBlock(embed_dim, hidden_dim, input_dim, num_layers_decoder, dropout)
  def forward(self, input):
    x, _ = self.encoder(input)
    x = x[-1, :, :].unsqueeze(0).repeat(input.shape[0], 1, 1)
    x, _ = self.decoder(x)
    return x

class GRUAutoEncoderAll(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers_encoder, num_layers_decoder, dropout=0.1):
      super().__init__()
      assert embed_dim < input_dim
      self.encoder = GRUBlock(input_dim, hidden_dim, embed_dim, num_layers_encoder, dropout)
      self.decoder = GRUBlock(embed_dim, hidden_dim, input_dim, num_layers_decoder, dropout)
    def forward(self, input):
      x, _ = self.encoder(x)
      x, _ = self.decoder(x)
      return x
