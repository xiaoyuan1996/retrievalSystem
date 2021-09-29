# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

# from model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math
from torch.autograd import Variable

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        out = self.fc(x)
        return self.linear(out)


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_k = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_q = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_merge = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C['fusion']['mca_HIDDEN_SIZE']
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C['fusion']['mca_HIDDEN_SIZE'],
            mid_size=__C['fusion']['mca_FF_SIZE'],
            out_size=__C['fusion']['mca_HIDDEN_SIZE'],
            dropout_r=__C['fusion']['mca_DROPOUT_R'],
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm1 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout2 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm2 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

    def forward(self, x, x_mask=None):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm1 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout2 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm2 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout3 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm3 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

    def forward(self, x, y, x_mask=None, y_mask=None):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x



if __name__ == "__main__":

    cfg = {"fusion":{
        'mca_HIDDEN_SIZE': 512,
        'mca_DROPOUT_R': 0.1,
        'mca_FF_SIZE': 1024,
        'mca_MULTI_HEAD': 8,
        'mca_HIDDEN_SIZE_HEAD': 64,
    }}


    self_att = SA(cfg)
    print(self_att)

    inputs = Variable(torch.zeros(10, 1, 512))
    inputs_l = Variable(torch.zeros(10, 1, 512))

    outputs = self_att(inputs)
    print(outputs.shape)

    guide_att = SGA(cfg)
    print(guide_att)
    outputs = guide_att(inputs, inputs_l)
    print(outputs.shape)