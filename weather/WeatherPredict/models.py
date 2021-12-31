# -*- coding:utf-8 -*-
"""
Writer: RuiStarlit
File: models
Project: HW
Create Time: 2021-12-27
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from WeatherPredict.attn import *


class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, enc_in=7, dec_in=1, label_len=8,
                 out_seq_len=1, n_encoder_layers=2, n_decoder_layers=2,
                 n_heads=4, dropout=0.1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_in
        # Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads, dropout))
        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads, dropout))
        self.pos = PositionalEncoding(dim_val)
        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(enc_in, dim_val)
        self.dec_input_fc = nn.Linear(dec_in, dim_val)
        self.out_fc = nn.Linear(dim_val, out_seq_len)

    def forward(self, x, y):
        # encoder
        e = self.encs[0](self.pos(self.enc_input_fc(x)))
        # e = self.encs[0](self.enc_input_fc(x))
        for enc in self.encs[1:]:
            e = enc(e)
        # decoder
        d = self.decs[0](self.dec_input_fc(y), e)
        for dec in self.decs[1:]:
            d = dec(d, e)
        # output
        x = self.out_fc(d.flatten(start_dim=1))

        return x


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
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


class LSTM(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads, dropout):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                           dropout=dropout, batch_first=True)

        self.fc9 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc10 = nn.Linear(dim_val * 2, 1)

    def forward(self, x):
        x, (hn, cn) = self.rnn(x)

        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        x = self.fc10(F.elu(self.fc9(hn)))
        return x


class AttentionLSTM(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads, dropout):
        super(AttentionLSTM, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.fc1 = nn.Linear(input_size, dim_val)

        self.attn = AttentionLayer(Attention(False), dim_val, n_heads, dim_attn, dim_attn)
        self.norm = nn.LayerNorm(dim_val)
        self.rnn = nn.LSTM(input_size=dim_val, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                           dropout=dropout, batch_first=True)

        self.fc2 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc3 = nn.Linear(dim_val * 2, 1)

    def forward(self, x):
        x = self.fc1(x)
        a = self.attn(x, x, x, None)
        x = self.norm(a + x)

        x, (hn, cn) = self.rnn(x)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        x = self.fc3(F.elu(self.fc2(hn)))
        return x


class LSTMAttetion(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads, dropout):
        super(LSTMAttetion, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.fc1 = nn.Linear(input_size, dim_val)

        self.attn = AttentionLayer(Attention(False), dim_val, n_heads, dim_attn, dim_attn)
        self.norm = nn.LayerNorm(dim_val)
        self.rnn = nn.LSTM(input_size=dim_val, hidden_size=dim_val, num_layers=n_layers, bidirectional=False,
                           dropout=dropout, batch_first=True)

        self.fc2 = nn.Linear(dim_val * 2, dim_val * 2)
        self.fc3 = nn.Linear(dim_val * 2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x, (hn, cn) = self.rnn(x)

        a = self.attn(x, x, x, None)

        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        hn = torch.cat((hn, a))
        x = self.fc3(F.elu(self.fc2(hn)))
        return x
