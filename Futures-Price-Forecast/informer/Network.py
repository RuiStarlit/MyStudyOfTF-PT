import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn , n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
    
    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)
        
        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + a)
        
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)
        
    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)
        
        a = self.attn2(x, kv = enc)
        x = self.norm2(a + x)
        
        a = self.fc1(F.elu(self.fc2(x)))
        
        x = self.norm3(x + a)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers = 1, n_encoder_layers = 1, n_heads = 1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len
        
        #Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads))
        
        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads))
        
        self.pos = PositionalEncoding(dim_val)
        
        #Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        # self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len * input_size)
        self.out_reg_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)
        self.out_seq_len = out_seq_len
        self.input_size = input_size

    def forward(self, x):
        #encoder
        e = self.encs[0](self.pos(self.enc_input_fc(x)))
        for enc in self.encs[1:]:
            e = enc(e)
        
        #decoder
        d = self.decs[0](self.dec_input_fc(x[:,-self.dec_seq_len:]), e)
        for dec in self.decs[1:]:
            d = dec(d, e)
        
        #output
        # x = self.out_fc(d.flatten(start_dim=1))
        x = self.out_reg_fc(d.flatten(start_dim=1))
        # x = torch.reshape(x, [-1, self.out_seq_len, self.input_size])
        return x

        
import torch.nn as nn
class classifier(nn.Module): 
    #定义所有层 
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout): 
        super().__init__() 
        #embedding 层 
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        #lstm 层 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True) 
        #全连接层 
        self.fc = nn.Linear(hidden_dim * 2, output_dim) 
        #激活函数 
        self.act = nn.Sigmoid() 
    def forward(self, embedded, text_lengths): 
        #text = [batch size,sent_length] 
        # embedded = self.embedding(text) 
        #embedded = [batch size, sent_len, emb dim] 
        packed_embedded = embedded #nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True) 
        packed_output, (hidden, cell) = self.lstm(packed_embedded) 
        print(embedded.shape, packed_embedded.data.shape, packed_output.data.shape, hidden.shape)
        #hidden = [batch size, num layers * num directions,hid dim] 
        #cell = [batch size, num layers * num directions,hid dim] 
        #连接最后的正向和反向隐状态 
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) 
        #hidden = [batch size, hid dim * num directions] 
        dense_outputs=self.fc(hidden) 
        #激活 
        outputs=self.act(dense_outputs) 
        return outputs
            
class CNN(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads):
        super(CNN, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val

        self.conv = nn.Conv2d(input_size, dim_val, (1, 1))

        self.conv1 = nn.Conv2d(input_size, dim_val, (3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(input_size, dim_val, (5, 1), padding=(2, 0))
        self.conv3 = nn.Conv2d(input_size, dim_val, (7, 1), padding=(3, 0))
   
        self.pool = nn.AdaptiveMaxPool2d((1 ,1))

        self.pool1 = nn.AdaptiveMaxPool2d((1 ,1))
        self.pool2 = nn.AdaptiveMaxPool2d((1 ,1))
        self.pool3 = nn.AdaptiveMaxPool2d((1 ,1))
        
        self.bn = nn.BatchNorm2d(dim_val)

        self.bn1 = nn.BatchNorm2d(dim_val)
        self.bn2 = nn.BatchNorm2d(dim_val)
        self.bn3 = nn.BatchNorm2d(dim_val)
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(dim_val*3, 1)
        self.fc1 = nn.Linear(dim_val*3, dim_val)
        self.fc2 = nn.Linear(dim_val, 1)

    def forward(self, x):
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        
        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        x3 = F.elu(self.conv3(x))
        
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.pool(x)
        
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        # x = self.fc2(F.tanh(self.fc1(x)))
        x = self.fc(x)
        return x
    
class RNN(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, n_layers, n_heads):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.dim_val = dim_val
        
        self.rnn = nn.LSTM(input_size = input_size, hidden_size =dim_val, num_layers =n_layers, bidirectional=False, dropout =0.2, batch_first=True)
        
        self.fc1 = nn.Linear(dim_val*2, dim_val*2)
        self.fc2 = nn.Linear(dim_val*2, 1)

    def forward(self, x):
        x, (hn, cn) = self.rnn(x)
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1) 
        x = self.fc2(F.elu(self.fc1(hn)))
        return x
if __name__ == '__main__':
    # print(classifier(231, 16, 5, 1,1,False,0.2)(torch.Tensor(np.random.randn(3, 20, 16)), torch.Tensor([20]*3)))
    print(RNN(16, 5, 15, 2, 3)(torch.Tensor(np.random.randn(3, 20, 15))))

        
        
        
        
        
        
        
        