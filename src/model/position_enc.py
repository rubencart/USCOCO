import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalSequenceEncoder(nn.Module):
    def __init__(self, input_feature_size, d_model, max_seq_len=80, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.feature_size = input_feature_size

        self.embeddings = nn.Linear(input_feature_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = self.embeddings(x)

        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(x.device)

        x = self.norm(x)
        x = self.dropout(x)
        return x

    def get_pos_emb(self, positions):
        seq_len = positions.size(1)
        pos = Variable(self.pe[:, :seq_len], requires_grad=False).to(positions.device) / math.sqrt(
            self.d_model
        )
        return pos


class PositionalTreeEncoder(nn.Module):
    def __init__(self, input_feature_size, d_model, n=2, k=32, p_repeats=32, dropout=0.1):
        # width n, depth k, embedding size d_model
        super().__init__()
        self.d_model = d_model
        self.max_n = n
        self.max_k = k

        self.p_repeats = p_repeats
        self.p = torch.nn.Parameter(
            torch.ones(self.p_repeats, dtype=torch.float32), requires_grad=True
        )
        self.init_weights()

        self.node_embeddings = nn.Linear(input_feature_size, d_model)
        self.pos_embeddings = nn.Linear(n * k * p_repeats, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        self.p.data.uniform_(0.7, 0.999)

    def get_weight_matrix(self):
        p = torch.tanh(self.p)
        p_weights = p.repeat_interleave(self.max_k * self.max_n)
        depths = (
            torch.arange(self.max_k)
            .repeat_interleave(self.max_n)
            .repeat(self.p_repeats)
            .to(p.device)
        )
        norm = torch.sqrt(1 - p_weights**2)
        scale = 1 / math.sqrt(self.d_model / 2)
        p_weights = torch.pow(p_weights, depths) * norm * scale
        return p_weights

    def forward(self, x, positions):
        node_emb = self.node_embeddings(x)
        pos_emb = self.get_pos_emb(positions)
        emb = self.norm(node_emb + pos_emb)
        emb = self.dropout(emb)
        return emb

    def get_pos_emb(self, positions):
        p_weights = self.get_weight_matrix()
        return self.pos_embeddings(positions.repeat((1, 1, self.p_repeats)) * p_weights)
