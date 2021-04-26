
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

class CNN_pre_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.conv12 = nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU()
        self.downsample1 = nn.MaxPool1d(2)
        self.conv21 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.relu21 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.conv22 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.relu22 = nn.ReLU()
        self.downsample2 = nn.MaxPool1d(2)
        self.net = nn.Sequential(self.conv11, self.relu11, self.dropout1, self.conv12, self.relu12, self.downsample1,
                                 self.conv21, self.relu21, self.dropout2, self.conv22, self.relu22,  self.downsample2)
        self.linear = nn.Linear(96, 24)
    def forward(self, input):
        seq_length, batch_size, _ = input.shape
        out = [self.net(input[i, :, :].unsqueeze(1)).reshape(batch_size, -1).unsqueeze(0) for i in range(seq_length)]
        out = torch.cat(out, axis=0)
        return self.linear(out)

class CNN_post_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(24, 96)
        self.conv11 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.conv12 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU()
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv21 = nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.relu21 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.conv22 = nn.Conv1d(8, 1, kernel_size=3, padding=1)
        self.relu22 = nn.ReLU()
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.net = nn.Sequential(self.upsample1, self.conv11, self.relu11, self.dropout1, self.conv12, self.relu12,
                                 self.upsample2, self.conv21, self.relu21, self.dropout2, self.conv22)
    def forward(self, input):
        input = self.linear(input)
        seq_length, batch_size, _ = input.shape
        out = [self.net(input[i, :, :].reshape(batch_size, 16, -1)).squeeze(1).unsqueeze(0) for i in range(seq_length)]
        out = torch.cat(out)
        return out

class CNNLSTMAutoEncoder(nn.Module):
    def __init__(self, step_size, input_dim, hidden_dim, embed_dim, num_layers_encoder, num_layers_decoder, layer_norm=False, dropout=0.1):
        super().__init__()
        self.CNN_pre_LSTM = CNN_pre_LSTM()
        self.encoder = LSTMBlock(step_size, hidden_dim, embed_dim, num_layers_encoder, layer_norm, dropout)
        self.decoder = LSTMBlock(embed_dim, hidden_dim, step_size, num_layers_decoder, layer_norm, dropout)
        self.CNN_post_LSTM = CNN_post_LSTM()
    def forward(self, input):
        input = self.CNN_pre_LSTM(input)
        x, _ = self.encoder(input)
        x = x[-1, :, :].unsqueeze(0).repeat(input.shape[0], 1, 1)
        x, _ = self.decoder(x)
        x = self.CNN_post_LSTM(x)
        return x
