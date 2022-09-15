# from https://github.com/jojonki/Gated-Convolutional-Networks
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GatedCNN(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 seq_len,
                 vocab_size,
                 embd_size,
                 n_layers,
                 kernel,
                 padding,
                 out_chs,
                 hidden_chs,
                 res_block_count,
                 init_factors_path,
                 cutoffs):
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count
        self.n_layers = n_layers

        
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(np.load(init_factors_path)[:,:embd_size]), freeze=False).float()

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv_0 = nn.Conv2d(1, out_chs, kernel, padding=padding)
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=padding)
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))

        self.down_conv = nn.ModuleList([nn.Conv2d(out_chs, hidden_chs, (kernel[0], 1), padding=padding) for _ in range(n_layers)])
        self.bottle_conv = nn.ModuleList([nn.Conv2d(hidden_chs, hidden_chs, (kernel[0], 1), padding=padding) for _ in range(n_layers)])  # bottleneck here
        self.up_conv = nn.ModuleList([nn.Conv2d(hidden_chs, out_chs, (kernel[0], 1), padding=padding) for _ in range(n_layers)])

        self.down_conv_gate = nn.ModuleList([nn.Conv2d(out_chs, hidden_chs, (kernel[0], 1), padding=padding) for _ in range(n_layers)]) # bottlenecking here
        self.bottle_conv_gate = nn.ModuleList([nn.Conv2d(hidden_chs, hidden_chs, (kernel[0], 1), padding=padding) for _ in range(n_layers)])
        self.up_conv_gate = nn.ModuleList([nn.Conv2d(hidden_chs, out_chs, (kernel[0], 1), padding=padding) for _ in range(n_layers)]) # bottlenecking here

        self.fc = nn.Linear(out_chs*(seq_len - 1), vocab_size)
        self.adapt = nn.AdaptiveLogSoftmaxWithLoss(out_chs*(seq_len - 1), vocab_size, cutoffs, div_value=1.8)
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])

    def forward(self, x):
        # x: (N, seq_len)

        # Embedding
        bs = x.size(0) # batch size
        if self.training:
          target = x[:,-1]
          x = x[:,:-1]
        seq_len = x.size(1)
        x = self.embedding(x) # (bs, seq_len, embd_size)

        # CNN
        x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        

        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x)      # (bs, Cout, seq_len, 1)
        A += self.b_0.repeat(1, 1, seq_len, 1)
        B = self.conv_gate_0(x) # (bs, Cout, seq_len, 1)
        B += self.c_0.repeat(1, 1, seq_len, 1)
        h = A * torch.sigmoid(B)    # (bs, Cout, seq_len, 1)
        res_input = h # TODO this is h1 not h0

        for i in range(self.n_layers):
            A = self.up_conv[i](self.bottle_conv[i](self.down_conv[i](h))) + self.b[i].repeat(1, 1, seq_len, 1)
            B = self.up_conv_gate[i](self.bottle_conv_gate[i](self.down_conv_gate[i](h))) + self.c[i].repeat(1, 1, seq_len, 1)
            h = A * torch.sigmoid(B) # (bs, Cout, seq_len, 1)
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h

        h = h.view(bs, -1) # (bs, Cout*seq_len)
        if self.training:
          return self.adapt(h, target)
        else :
          return self.adapt.log_prob(h)

class RegressionGatedCNN(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 vocab_size,
                 embd_size,
                 n_layers,
                 kernel,
                 out_chs,
                 res_block_count,
                 init_factors_path,
                 k_pool=3,
                 drop_p=0.0):
        super(RegressionGatedCNN, self).__init__()
        self.res_block_count = res_block_count
        self.n_layers = n_layers
        self.k_pool = k_pool
        enhanced_embedding = np.zeros((vocab_size, embd_size))
        enhanced_embedding[1:, :] = np.load(init_factors_path)[:,:embd_size]
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(enhanced_embedding), freeze=False).float()
        self.padding_0 = nn.ConstantPad2d((0, 0, kernel[0] - 1, 0), 0) # not using future songs to predict current songs
        self.conv_0 = nn.Conv2d(1, out_chs, kernel)
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1))
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel)
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1))
        self.batch_norm_0 = nn.BatchNorm1d(out_chs)
        self.relu_0 = nn.ReLU()
        self.drop_layer_0 = nn.Dropout(p=drop_p)

        self.paddings = nn.ModuleList([nn.ConstantPad1d((kernel[0] - 1, 0), 0) for _ in range(n_layers)])  # not using future songs to predict current songs
        self.bottle_conv = nn.ModuleList([nn.Conv1d(out_chs, out_chs, kernel[0]) for _ in range(n_layers)]) 
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1)) for _ in range(n_layers)])
        self.bottle_conv_gate = nn.ModuleList([nn.Conv1d(out_chs, out_chs, kernel[0]) for _ in range(n_layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1)) for _ in range(n_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(out_chs) for _ in range(n_layers)])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(n_layers)])
        self.drop_layers = nn.ModuleList([nn.Dropout(p=drop_p) for _ in range(n_layers)])

        self.fc = nn.Linear(out_chs * self.k_pool, embd_size) # the regression model outputs an embedding
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in')

    def forward(self, x):
        # x: (N, seq_len)

        # Embedding
        bs = x.size(0) # batch size
        seq_len = x.size(1)
        x = self.embedding(x) # (bs, seq_len, embd_size)

        # CNN
        x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        x = self.padding_0(x)

        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x).squeeze(3)      # (bs, Cout, seq_len)
        A += self.b_0.repeat(1, 1, seq_len)

        B = self.conv_gate_0(x).squeeze(3) # (bs, Cout, seq_len)
        B += self.c_0.repeat(1, 1, seq_len)
        h = A * torch.sigmoid(B)    # (bs, Cout, seq_len)
        h = self.batch_norm_0(h)
        #h = self.relu_0(h)
        h = self.drop_layer_0(h)
        res_input = h # TODO this is h1 not h0

        for i in range(self.n_layers):
            h = self.paddings[i](h)
            A = self.bottle_conv[i](h) # + self.b[i].repeat(1, 1, seq_len)
            A += self.b[i]
            B = self.bottle_conv_gate[i](h) # + self.c[i].repeat(1, 1, seq_len)
            B += self.c[i]
            h = A * torch.sigmoid(B) # (bs, Cout, seq_len)
            h = self.batch_norms[i](h)
            #h = self.relus[i](h)
            h = self.drop_layers[i](h)
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h
        h =  torch.topk(h, k =self.k_pool, dim=2)[0]
        h = h.view(bs, -1) # (bs, Cout*seq_len)
        h = self.fc(h)
        return h