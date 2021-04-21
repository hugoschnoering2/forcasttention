
import torch
import torch.nn as nn

class GRU_AE(nn.Module):

  def __init__(self, input_dim, hidden_dim, num_layers_encoder, num_layers_decoder):
    super().__init__()

    self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers_encoder)
    self.decoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers_decoder)
    self.activation = nn.ReLU()
    self.linear = nn.Linear(in_features=hidden_dim, out_features=input_dim)

  def forward(self, x):
    """
    x : input tensor of dim (seq_len, batch_size, input_dim)
    """
    out, _ = self.encoder(x)
    out = out[-1, :, :].unsqueeze(0).repeat(x.shape[0], 1, 1)
    out, _ = self.decoder(out)
    pred_x = self.linear(self.activation(out))
    return pred_x


class GRU_AE_all(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers_encoder, num_layers_decoder):
      super().__init__()
      assert hidden_dim < input_dim, "there is basically no summurization"
      self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers_encoder)
      self.decoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers_decoder)
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
