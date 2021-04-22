
import torch
import torch.nn as nn

from models.tcn.tcn import Chomp1d


class Chomp1d(nn.Module):
    """
    source for this class : https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DownCNNBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, dilation, down_factor, res):
        super().__init__()

        self.res = res
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.downsample = nn.MaxPool1d(kernel_size=down_factor)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)

        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1) if (input_dim != output_dim and res) else None

    def forward(self, x):
        y = self.net(x)
        if self.proj is None and self.res:
            y = torch.relu(x+y)
        elif self.proj is not None:
            y = torch.relu(self.proj(x)+y)
        else:
            pass
        return self.downsample(y)


class UpCNNBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, dilation, up_factor, res):
        super().__init__()

        self.res = res
        padding = (kernel_size - 1) * dilation // 2
        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size, dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(input_dim, output_dim, kernel_size, dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)

        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1) if (input_dim != output_dim and res) else None

    def forward(self, x):
        x = self.upsample(x)
        y = self.net(x)
        if self.proj is None and self.res:
            y = torch.relu(x+y)
        elif self.proj is not None:
            y = torch.relu(self.proj(x)+y)
        else:
            pass
        return y

class DownTCNBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, dilation, down_factor, res):
        super().__init__()

        self.res = res
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.downsample = nn.MaxPool1d(kernel_size=down_factor)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1) if (input_dim != output_dim and res) else None

    def forward(self, x):
        y = self.net(x)
        if self.proj is None and self.res:
            y = torch.relu(x+y)
        elif self.proj is not None:
            y = torch.relu(self.proj(x)+y)
        else:
            pass
        return self.downsample(y)


class UpTCNBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, dilation, up_factor, res):
        super().__init__()

        self.res = res
        padding = (kernel_size - 1) * dilation
        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(input_dim, output_dim, kernel_size, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1) if (input_dim != output_dim and res) else None

    def forward(self, x):
        x = self.upsample(x)
        y = self.net(x)
        if self.proj is None and self.res:
            y = torch.relu(x+y)
        elif self.proj is not None:
            y = torch.relu(self.proj(x)+y)
        else:
            pass
        return y



class CNN_AE_u(nn.Module):

    def __init__(self, model, input_dim, base_dim, depth, kernel_size,
                 scale_factor, embedding_size, dilation, res):

        super().__init__()

        tcn = model == "tcn_ae"
        if tcn :
            padding = (kernel_size-1) * dilation
            down_block = DownTCNBlock
            up_block = UpTCNBlock
        else:
            padding = (kernel_size-1) * dilation //2
            down_block = DownCNNBlock
            up_block = UpCNNBlock

        self.base_dim = base_dim
        self.depth = depth

        encoder = []
        prev_dim = input_dim[0]
        for i in range(0, depth):
            next_dim = base_dim if i == 0 else prev_dim * 2
            encoder.append(down_block(prev_dim, next_dim, kernel_size, dilation=dilation, down_factor=scale_factor, res=res))
            prev_dim = next_dim
        if tcn:
            encoder.extend([nn.Conv1d(prev_dim, prev_dim, kernel_size, dilation=dilation, padding=padding),
                            Chomp1d(padding), nn.ReLU(), nn.Dropout(0.1)])
        else:
            encoder.extend([nn.Conv1d(prev_dim, prev_dim, kernel_size, dilation=dilation, padding=padding),
                            nn.ReLU(), nn.Dropout(0.1)])
        self.encoder = nn.Sequential(*encoder)


        self.linear_enc = nn.Linear(in_features=int(prev_dim*input_dim[1]/scale_factor**depth), out_features=embedding_size)
        self.linear_dec = nn.Linear(in_features=embedding_size, out_features=int(prev_dim*input_dim[1]/scale_factor**depth))


        if tcn:
            decoder = [nn.Conv1d(prev_dim, prev_dim, kernel_size, dilation=dilation, padding=padding),
                       Chomp1d(padding), nn.ReLU(), nn.Dropout(0.1), nn.ReLU(), nn.Dropout(0.1)]
        else:
            decoder = [nn.Conv1d(prev_dim, prev_dim, kernel_size, dilation=dilation, padding=padding),
                       nn.ReLU(), nn.Dropout(0.1), nn.ReLU(), nn.Dropout(0.1)]
        for i in range(depth-1):
            next_dim = prev_dim // 2
            decoder.append(up_block(prev_dim, next_dim, kernel_size, dilation=dilation, up_factor=scale_factor, res=res))
            prev_dim = next_dim
        if tcn:
            decoder.extend([nn.Upsample(scale_factor=scale_factor, mode="nearest"),
                            nn.Conv1d(prev_dim, prev_dim, kernel_size, dilation=dilation, padding=padding), Chomp1d(padding), nn.ReLU(), nn.Dropout(0.1),
                            nn.Conv1d(prev_dim, input_dim[0], kernel_size, dilation=dilation, padding=padding), Chomp1d(padding)])
        else:
            decoder.extend([nn.Upsample(scale_factor=scale_factor, mode="nearest"),
                            nn.Conv1d(prev_dim, prev_dim, kernel_size, dilation=dilation, padding=padding), nn.ReLU(), nn.Dropout(0.1),
                            nn.Conv1d(prev_dim, input_dim[0], kernel_size, dilation=dilation, padding=padding)])
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):

        bs = x.shape[0]
        e = self.encoder(x)
        e = self.linear_enc(e.reshape(bs, -1))
        e = self.linear_dec(e)
        pred_x = self.decoder(e.reshape(bs, self.base_dim*2**(self.depth-1), -1))
        return pred_x
