
import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    """
    this class comes from : https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class DownCNNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation, down_factor, res=False, dropout=0.1):
        super().__init__()
        self.res = res
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.downsample = nn.MaxPool1d(kernel_size=down_factor) if down_factor > 1 else None
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1) if (input_dim != output_dim and res) else None
    def forward(self, input):
        x = self.net(input)
        if self.proj is None and self.res:
            x = torch.relu(x+input)
        elif self.proj is not None:
            x = torch.relu(self.proj(input)+x)
        else:
            pass
        if self.downsample is not None:
            return self.downsample(x)
        return x

class UpCNNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation, up_factor, res=False, dropout=0.1):
        super().__init__()
        self.res = res
        padding = (kernel_size - 1) * dilation // 2
        self.upsample = nn.Upsample(scale_factor=up_factor) if up_factor > 1 else None
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size, dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(input_dim, output_dim, kernel_size, dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1) if (input_dim != output_dim and res) else None
    def forward(self, input):
        if self.upsample is not None:
            input = self.upsample(input)
        x = self.net(input)
        if self.proj is None and self.res:
            x = torch.relu(input+x)
        elif self.proj is not None:
            x = torch.relu(self.proj(input)+x)
        else:
            pass
        return x

class DownTCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation, down_factor, res=False, dropout=0.1):
        super().__init__()
        self.res = res
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.downsample = nn.MaxPool1d(kernel_size=down_factor) if down_factor > 1 else None
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1) if (input_dim != output_dim and res) else None
    def forward(self, input):
        x = self.net(input)
        if self.proj is None and self.res:
            x = torch.relu(x+input)
        elif self.proj is not None:
            x = torch.relu(x+self.proj(input))
        else:
            pass
        if self.downsample is not None:
            return self.downsample(x)
        return x


class UpTCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation, up_factor, res=False, dropout=0.1):
        super().__init__()
        self.res = res
        padding = (kernel_size - 1) * dilation
        self.upsample = nn.Upsample(scale_factor=up_factor) if up_factor > 1 else None
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(input_dim, output_dim, kernel_size, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1) if (input_dim != output_dim and res) else None
    def forward(self, input):
        if self.upsample is not None:
            input = self.upsample(input)
        x = self.net(input)
        if self.proj is None and self.res:
            x = torch.relu(input+x)
        elif self.proj is not None:
            x = torch.relu(self.proj(input)+x)
        else:
            pass
        return x

class CNNEncoder(nn.Module):
    def __init__(self, model, input_dim, base_dim, depth, kernel_size, scale_factor,
                 embedding_size, dilation_factor, res=False, dropout=0.1):
        super().__init__()
        tcn = model == "tcn_ae"
        down_block = DownTCNBlock if tcn else DownCNNBlock
        encoder = []
        prev_dim = input_dim[0]
        for i in range(0, depth):
            next_dim = base_dim if i == 0 else prev_dim * 2
            dilation = dilation_factor ** i
            encoder.append(down_block(prev_dim, next_dim, kernel_size, dilation=dilation, down_factor=scale_factor, res=res, dropout=dropout))
            prev_dim = next_dim
        if tcn:
            encoder.extend([nn.Conv1d(prev_dim, prev_dim, kernel_size, dilation=dilation, padding=(kernel_size-1) * dilation),
                            Chomp1d((kernel_size-1) * dilation)), nn.ReLU(), nn.Dropout(p=dropout)])
        else:
            encoder.extend([nn.Conv1d(prev_dim, prev_dim, kernel_size, dilation=dilation, padding=(kernel_size-1) * dilation // 2),
                            nn.ReLU(), nn.Dropout(p=dropout)])
        self.linear = nn.Linear(in_features=int(prev_dim*input_dim[1]/scale_factor**depth), out_features=embedding_size)
        self.net = nn.Sequential(*encoder)
    def forward(self, input):
        x = self.net(input)
        x = self.linear(x.reshape(input.shape[0], -1))
        return x

class CNNDecoder(nn.Module):
    def __init__(self, model, input_dim, base_dim, depth, kernel_size, scale_factor,
                 embedding_size, dilation_factor, res=False, dropout=0.1):
        super().__init__()
        self.base_dim = base_dim
        self.depth = depth
        tcn = model == "tcn_ae"
        up_block = UpTCNBlock if tcn else UpCNNBlock
        prev_dim = base_dim*2**(depth-1)
        self.linear = nn.Linear(in_features=embedding_size, out_features=int(prev_dim*input_dim[1]/scale_factor**depth))
        dilation = dilation_factor ** (depth-1)
        padding = (kernel_size - 1) * dilation
        if tcn:
            decoder = [nn.Conv1d(prev_dim, prev_dim, kernel_size, dilation=dilation, padding=padding),
                       Chomp1d(padding), nn.ReLU(), nn.Dropout(p=dropout)]
        else:
            padding = padding // 2
            decoder = [nn.Conv1d(prev_dim, prev_dim, kernel_size, dilation=dilation, padding=padding),
                       nn.ReLU(), nn.Dropout(p=dropout)]
        for i in range(depth-1):
            next_dim = prev_dim // 2
            decoder.append(up_block(prev_dim, next_dim, kernel_size, dilation=dilation, up_factor=scale_factor, res=res, dropout=dropout))
            dilation = dilation // dilation_factor
            prev_dim = next_dim
        if scale_factor > 1:
            decoder.append(nn.Upsample(scale_factor=scale_factor, mode="nearest"))
        if tcn:
            padding = (kernel_size-1)
            decoder.extend([nn.Conv1d(prev_dim, prev_dim, kernel_size, padding=padding), Chomp1d(padding), nn.ReLU(), nn.Dropout(p=dropout),
                            nn.Conv1d(prev_dim, input_dim[0], kernel_size, padding=padding), Chomp1d(padding)])
        else:
            padding = (kernel_size-1) // 2
            decoder.extend([nn.Conv1d(prev_dim, prev_dim, kernel_size, padding=padding), nn.ReLU(), nn.Dropout(p=dropout),
                            nn.Conv1d(prev_dim, input_dim[0], kernel_size, padding=padding)])
        self.net = nn.Sequential(*decoder)
    def forward(self, input):
        x = self.linear(input).reshape(input.shape[0], self.base_dim*2**(self.depth-1), -1)
        x = self.net(x)
        return x

class CNNAutoEncoder(nn.Module):
    def __init__(self, model, input_dim, base_dim, depth, kernel_size, scale_factor,
                 embedding_size, dilation_factor, res=False, dropout=0.1):
        super().__init__()
        self.encoder = CNNEncoder(model, input_dim, base_dim, depth, kernel_size, scale_factor,
                                  embedding_size, dilation_factor, res, dropout)
        self.decoder = CNNDecoder(model, input_dim, base_dim, depth, kernel_size, scale_factor,
                                  embedding_size, dilation_factor, res, dropout)
    def forward(self, input):
        emb = self.encoder(input)
        out = self.decoder(emb)
        return out
