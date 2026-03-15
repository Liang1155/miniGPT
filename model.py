import torch
import math
import torch.nn as nn
import torch.nn.functional as F



class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)


        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0)*torch.arange(0,d_model,2).float()/d_model)
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe',pe)


    def forward(self,x):
        x = x + self.pe[:x.size(0),:]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        score = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask==0,-1e9)

        attn_weights = F.softmax(score,-1)
        output = torch.matmul(attn_weights,V)
        return output,attn_weights

    def split_heads(self, x):

        batch_size,seq_len,_ = x.size()
        x = x.view(batch_size,seq_len,self.num_heads,self.d_k)
        return x.transpose(2,1)

    def combine_heads(self, x):

        batch_size,_,seq_len,_ = x.size()
        x = x.transpose(1,2).contiguous()
        x = x.view(batch_size,seq_len,self.d_model)
        return x


    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.combine_heads(attn_output))
        return output


class FeedForward(nn.Module):

    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self,d_model,num_heads,d_ff,dropout=0.1,):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model,num_heads)
        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, tgt_mask=None):
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x,tgt_mask=None):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(
        self,
        tgt_vocab_size,   # 目标语言词表大小
        d_model=512,      # 模型维度（论文默认512）
        num_heads=8,      # 注意力头数（论文默认8）
        num_layers=6,     # Encoder/Decoder 层数（论文默认6）
        d_ff=2048,        # FFN 中间层维度（论文默认2048）
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_tgt_mask(self, tgt, pad_idx=0):

        tgt_len = tgt.size(1)
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]

        # 下三角矩阵：位置 i 只能看到 0..i
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)    # [1, 1, tgt_len, tgt_len]

        return pad_mask & causal_mask

    def forward(self, tgt, pad_idx=0):

        tgt_mask = self.make_tgt_mask(tgt, pad_idx)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_emb = self.pos_encoding(tgt_emb).transpose(0, 1)
              # [batch, src_len, d_model]

        # 解码
        dec_output = self.decoder(tgt_emb, tgt_mask)  # [batch, tgt_len, d_model]

        # 输出层：每个位置预测下一个词的概率
        logits = self.output_linear(dec_output)                        # [batch, tgt_len, tgt_vocab_size]
        return logits