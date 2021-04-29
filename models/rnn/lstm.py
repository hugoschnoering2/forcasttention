
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
    def __init__(self, step_size, input_dim):
        super().__init__()

        self.conv11 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.dropout11 = nn.Dropout(0.1)
        self.conv12 = nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU()
        self.dropout12 = nn.Dropout(0.1)
        self.block1 = nn.Sequential(self.conv11, self.relu11, self.dropout11,
                                    self.conv12, self.relu12, self.dropout12)
        self.res_conv1 = nn.Conv1d(1, 8, kernel_size=1)
        self.downsample1 = nn.MaxPool1d(2)

        self.conv21 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.relu21 = nn.ReLU()
        self.dropout21 = nn.Dropout(0.1)
        self.conv22 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.relu22 = nn.ReLU()
        self.dropout22 = nn.Dropout(0.1)
        self.block2 = nn.Sequential(self.conv21, self.relu21, self.dropout21,
                                    self.conv22, self.relu22, self.dropout22)
        self.res_conv2 = nn.Conv1d(8, 16, kernel_size=1)
        self.downsample2 = nn.MaxPool1d(2)

        self.conv31 = nn.Conv1d(16, 16, kernel_size=1)
        self.relu31 = nn.ReLU()
        self.dropout31 = nn.Dropout(0.1)
        self.block3 = nn.Sequential(self.conv31, self.relu31, self.dropout31)

        self.linear = nn.Linear(step_size * 4, input_dim)

    def forward_(self, input):
        x1 = torch.relu(self.block1(input) + self.res_conv1(input))
        x1 = self.downsample1(x1)
        x2 = torch.relu(self.block2(x1) + self.res_conv2(x1))
        x2 = self.downsample2(x2)
        x3 = self.block3(x2)
        return x3

    def forward(self, input):
        seq_length, batch_size, _ = input.shape
        out = [self.forward_(input[i, :, :].unsqueeze(1)).reshape(batch_size, -1).unsqueeze(0) for i in range(seq_length)]
        out = torch.cat(out, axis=0)
        return self.linear(out)

class CNN_post_LSTM(nn.Module):
    def __init__(self, step_size, input_dim):
        super().__init__()

        self.linear = nn.Linear(input_dim, step_size * 4)

        self.conv01 = nn.Conv1d(16, 16, kernel_size=1)
        self.relu01 = nn.ReLU()
        self.dropout01 = nn.Dropout(0.1)
        self.block0 = nn.Sequential(self.conv01, self.relu01, self.dropout01)

        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv11 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.dropout11 = nn.Dropout(0.1)
        self.conv12 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU()
        self.dropout12 = nn.Dropout(0.1)
        self.block1 = nn.Sequential(self.conv11, self.relu11, self.dropout11,
                                    self.conv12, self.relu12, self.dropout12,)
        self.res_conv1 = nn.Conv1d(16, 8, kernel_size=1)

        self.upsample2 = nn.Upsample(scale_factor=2)

        self.conv21 = nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.relu21 = nn.ReLU()
        self.dropout21 = nn.Dropout(0.1)
        self.conv22 = nn.Conv1d(8, 1, kernel_size=3, padding=1)
        self.block2 = nn.Sequential(self.conv21, self.relu21, self.dropout21,
                                    self.conv22)

    def forward_(self, input):
        x1 = self.block0(input)
        x1 = self.upsample1(x1)
        x2 = torch.relu(self.block1(x1) + self.res_conv1(x1))
        x2 = self.upsample2(x2)
        x3 = self.block2(x2)
        return x3

    def forward(self, input):
        input = self.linear(input)
        seq_length, batch_size, _ = input.shape
        out = [self.forward_(input[i, :, :].reshape(batch_size, 16, -1)).squeeze(1).unsqueeze(0) for i in range(seq_length)]
        out = torch.cat(out)
        return out

class CNN_autoencoder(nn.Module):

    def __init__(self, step_size, input_dim):
        super().__init__()
        self.encoder = CNN_pre_LSTM(step_size, input_dim)
        self.decoder = CNN_post_LSTM(step_size, input_dim)
    def forward(self, input):
        emb = self.encoder(input)
        out = self.decoder(emb)
        return out

class CNNLSTMAutoEncoder(nn.Module):
    def __init__(self, step_size, input_dim, hidden_dim, embed_dim, num_layers_encoder, num_layers_decoder, layer_norm=False, dropout=0.1):
        super().__init__()
        self.CNN_pre_LSTM = CNN_pre_LSTM(step_size, input_dim)
        self.encoder = LSTMBlock(input_dim, hidden_dim, embed_dim, num_layers_encoder, layer_norm, dropout)
        self.decoder = LSTMBlock(embed_dim, hidden_dim, input_dim, num_layers_decoder, layer_norm, dropout)
        self.CNN_post_LSTM = CNN_post_LSTM(step_size, input_dim)
    def forward(self, input):
        input = self.CNN_pre_LSTM(input)
        x, _ = self.encoder(input)
        x = x[-1, :, :].unsqueeze(0).repeat(input.shape[0], 1, 1)
        x, _ = self.decoder(x)
        x = self.CNN_post_LSTM(x)
        return x
