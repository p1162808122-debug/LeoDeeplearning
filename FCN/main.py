# transformer_from_scratch.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Multi-Head Attention
# ---------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query/key/value: (batch, seq_len, d_model)
        B = query.size(0)

        # linear and split into heads
        def shape(x):
            # (B, seq_len, num_heads, d_k) -> (B, num_heads, seq_len, d_k)
            return x.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        q = shape(self.q_linear(query))
        k = shape(self.k_linear(key))
        v = shape(self.v_linear(value))

        # scaled dot-product attention
        # scores: (B, num_heads, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask expected shape broadcastable to (B, num_heads, q_len, k_len)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # context: (B, num_heads, q_len, d_k) -> concat -> (B, q_len, d_model)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.out_linear(context)
        return out, attn  # return attention if needed


# ---------------------------
# Position-wise Feed-Forward
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant positional encoding matrix with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return x


# ---------------------------
# Encoder Layer
# ---------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # self-attention
        attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # feed-forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


# ---------------------------
# Decoder Layer
# ---------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # masked self-attention
        self_attn_out, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        # encoder-decoder cross attention
        cross_out, _ = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = x + self.dropout(cross_out)
        x = self.norm2(x)

        # feed-forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x


# ---------------------------
# Encoder & Decoder stacks
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        # src: (batch, src_len) token ids
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)  # (batch, src_len, d_model)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        x = self.norm(x)
        logits = self.out_proj(x)  # (batch, tgt_len, vocab_size)
        return logits


# ---------------------------
# Full Transformer
# ---------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, num_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, num_heads, d_ff, dropout)

    def make_src_mask(self, src, pad_idx=0):
        # src: (B, src_len)
        # mask: (B, 1, 1, src_len) or (B, 1, src_len) - broadcastable
        mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,src_len)
        return mask  # 1 means token, 0 means pad

    def make_tgt_mask(self, tgt, pad_idx=0):
        # subsequent mask + padding mask
        B, tgt_len = tgt.size()
        subsequent = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()  # upper triangular
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,tgt_len)
        # combine: pad_mask & (~subsequent) -> but need broadcast shapes; we'll build final mask when used
        # We'll use mask as (B, num_heads, tgt_len, tgt_len) compatible with attention
        return pad_mask, subsequent  # return components so caller can combine if needed

    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        # src: (B, src_len), tgt: (B, tgt_len)
        src_mask = self.make_src_mask(src, pad_idx=src_pad_idx)  # (B,1,1,src_len)
        pad_mask_tgt, subsequent = self.make_tgt_mask(tgt, pad_idx=tgt_pad_idx)  # pad_mask_tgt: (B,1,1,tgt_len)
        # Create tgt_mask in attention shape (B,1,tgt_len,tgt_len) with subsequent masking
        # We want positions allowed=1, disallowed=0
        tgt_mask = pad_mask_tgt & (~subsequent.unsqueeze(0).unsqueeze(1))  # broadcast subsequent
        # encoder output
        memory = self.encoder(src, src_mask)
        logits = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return logits  # (B, tgt_len, tgt_vocab)


# ---------------------------
# Simple usage / toy training loop skeleton
# ---------------------------
if __name__ == "__main__":
    # toy config
    SRC_VOCAB = 1000
    TGT_VOCAB = 1000
    BATCH = 2
    SRC_LEN = 10
    TGT_LEN = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(src_vocab=SRC_VOCAB, tgt_vocab=TGT_VOCAB, d_model=128, N=2, num_heads=4, d_ff=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # assume pad index = 0

    # fake batch of token ids
    src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN)).to(device)
    tgt_in = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN)).to(device)  # input to decoder
    tgt_out = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN)).to(device)  # ground truth

    model.train()
    logits = model(src, tgt_in)  # (B, tgt_len, vocab)
    # reshape for loss: (B*tgt_len, vocab)
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = tgt_out.view(-1)
    loss = loss_fn(logits_flat, target_flat)

    loss.backward()
    optimizer.step()

    print("forward ok, loss:", loss.item())
