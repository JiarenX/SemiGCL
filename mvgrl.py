import torch
import torch.nn as nn


class MVGRL(nn.Module):
    def __init__(self, n_h):
        super(MVGRL, self).__init__()
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, seq3, seq4, msk=None, samp_bias1=None, samp_bias2=None):
        seq1 = seq1.unsqueeze(0)
        c_1 = self.read(seq1, msk)
        c_1 = self.sigm(c_1)

        seq2 = seq2.unsqueeze(0)
        c_2 = self.read(seq2, msk)
        c_2 = self.sigm(c_2)

        seq3 = seq3.unsqueeze(0)
        seq4 = seq4.unsqueeze(0)

        ret = self.disc(c_1, c_2, seq1, seq2, seq3, seq4, samp_bias1, samp_bias2)

        return ret


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, seq1, seq2, seq3, seq4, s_bias1=None, s_bias2=None):
        c_x_1 = torch.unsqueeze(c1, 1)
        c_x_1 = c_x_1.expand_as(seq1)
        c_x_2 = torch.unsqueeze(c2, 1)
        c_x_2 = c_x_2.expand_as(seq2)
        sc_1 = torch.squeeze(self.f_k(seq2, c_x_1), 2)
        sc_2 = torch.squeeze(self.f_k(seq1, c_x_2), 2)
        sc_3 = torch.squeeze(self.f_k(seq4, c_x_1), 2)
        sc_4 = torch.squeeze(self.f_k(seq3, c_x_2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)
