# -*- coding:utf-8 -*-
"""
Writer: RuiStarlit
File: model
Project: informer
Create Time: 2021-11-30
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils1 import *
from attn import *



class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, enc_in, dec_in, label_len,
                 out_seq_len=1, n_encoder_layers=1, n_decoder_layers=1,
                 n_heads=1, dropout=0.1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_in

        # Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads, dropout))

        self.rnn = nn.LSTM(input_size=dec_in, hidden_size=dec_in, num_layers=2, bidirectional=False,
                           dropout=dropout, batch_first=True)

        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads, dropout))

        self.pos = PositionalEncoding(dim_val)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(enc_in, dim_val)
        self.dec_input_fc = nn.Linear(dec_in, dim_val)
        self.out_fc1 = nn.Linear(label_len * dim_val, 2 * dim_val)
        # self.fc1 = nn.Linear(2 * dim_val, dim_val)
        self.out_fc2 = nn.Linear(4 * dim_val, dim_val)
        self.out_fc3 = nn.Linear(3 * dim_val + 4, dim_val)
        self.out_fc4 = nn.Linear(dim_val, out_seq_len)
        self.binary_fc = nn.Linear(3 * dim_val + 4, dim_val)
        self.binary_fc2 = nn.Linear(dim_val, 1)

    def forward(self, x, y):
        # encoder
        e = self.encs[0](self.pos(self.enc_input_fc(x)))
        # e = self.encs[0](self.enc_input_fc(x))
        # # #         print('e1',e.shape)
        for enc in self.encs[1:]:
            e = enc(e)
        # e, (hn, cn) = self.rnn(x)
        # print('e2',e.shape)
        # hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        # hn = F.elu(self.fc1(hn))

        # decoder
        # y [B, label_len, 2]
        y, (hn, cn) = self.rnn(y)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        d = self.decs[0](self.dec_input_fc(y), e)
        # d = self.decs[0](self.pos(self.dec_input_fc(y)), e)
        # #         print('d1',d.shape)
        for dec in self.decs[1:]:
            d = dec(d, e)

        # output
        x = F.elu(self.out_fc1(d.flatten(start_dim=1)))
        s = torch.cat((e[:, -1], x, hn), dim=1)
        # x = self.out_fc2(x)
        x = F.elu(self.out_fc3(s))
        x = self.out_fc4(x)

        # xx = self.binary_fc2(F.elu(self.binary_fc(s)))
        # xx = F.sigmoid(xx)

        return x


class Transformer_LSTM(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, enc_in, dec_in, label_len,
                 out_seq_len=1, n_encoder_layers=1, n_decoder_layers=1,
                 n_heads=1, dropout=0.1):
        super(Transformer_LSTM, self).__init__()
        self.dec_seq_len = dec_in

        self.rnn = nn.LSTM(input_size=enc_in, hidden_size=dim_val, num_layers=n_encoder_layers, bidirectional=False,
                           dropout=dropout, batch_first=True)

        self.decoder = nn.LSTM(input_size=dec_in, hidden_size=dim_val, num_layers=n_encoder_layers, bidirectional=False,
                           dropout=dropout, batch_first=True)

        self.attn = AttentionLayer(Attention(False), dim_val, n_heads, dim_attn, dim_attn)
        self.norm = nn.LayerNorm(dim_val)
        self.d_fc1 = nn.Linear(label_len * dim_val, 2* dim_val)
        self.d_fc2 = nn.Linear(2 * dim_val, 1 * dim_val)
        self.out_fc1 = nn.Linear(5 * dim_val, 2* dim_val)
        self.out_fc2 = nn.Linear(2 * dim_val, 2 * dim_val)

        self.out_fc3 = nn.Linear(2 * dim_val, dim_val)
        self.out_fc4 = nn.Linear(dim_val, out_seq_len)

    def forward(self, x, y):
        e, (hn_e, cn_e) = self.rnn(x)
        hn_e = torch.cat((hn_e[-2, :, :], hn_e[-1, :, :]), dim=1)

        d, (hn_d, cn_d) = self.decoder(y)
        hn_d = torch.cat((hn_d[-2, :, :], hn_d[-1, :, :]), dim=1)

        a = self.attn(d, e, e, None)
        d = self.norm(d + a)

        d = F.elu(self.d_fc1(d.flatten(start_dim=1)))
        d = self.d_fc2(d)
        d = torch.cat((hn_e, hn_d, d), dim=1)

        x = self.out_fc2(F.elu(self.out_fc1(d)))
        x = F.relu(x)

        x = self.out_fc4(F.elu(self.out_fc3(x)))
        # xx = F.sigmoid(xx)

        return x




class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn = AttentionLayer(Attention(False), dim_val, n_heads, dim_attn, dim_attn)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a = self.attn(x, x, x, None)
        x = self.norm1(x + a)

        a = self.fc1(F.elu(self.fc2(x)))
        a = self.dropout(a)
        x = self.norm2(x + a)

        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        # self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn1 = AttentionLayer(Attention(False), dim_val, n_heads, dim_attn, dim_attn)
        self.attn2 = AttentionLayer(Attention(False), dim_val, n_heads, dim_attn, dim_attn)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc):
        a = self.attn1(x, x, x, None)
        x = self.norm1(a + x)

        a = self.attn2(x, enc, enc, None)
        x = self.norm2(a + x)

        a = self.fc1(F.elu(self.fc2(x)))
        a = self.dropout(a)

        x = self.norm3(x + a)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x