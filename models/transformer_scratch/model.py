import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention module with scaled dot-product:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        batch_size = q.size(0)

        # Linear projections & split into n_heads
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        q_proj = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product scores: [batch_size, n_heads, q_len, k_len]
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Mask should broadcast to [batch_size, n_heads, q_len, k_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: [batch_size, n_heads, q_len, d_k]
        context = torch.matmul(attn_weights, v_proj)

        # Concatenate heads back to [batch_size, q_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer: Self-Attention + FeedForward with Add & LayerNorm
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None):
        # Self-attention sublayer with residual connection
        attn_out, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_out))
        # Feed-forward sublayer with residual connection
        ffn_out = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_out))
        return src

class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer: Masked Self-Attention + Cross-Attention + FeedForward
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        # 1. Masked self-attention
        attn_out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_out))
        
        # 2. Encoder-Decoder cross-attention
        cross_out, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_out))
        
        # 3. Position-wise feed-forward
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))
        return tgt

class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return tgt

class BasicTransformer(nn.Module):
    """
    Full Encoder-Decoder Transformer model for English -> Hindi translation.
    Trained from scratch.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # Token embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # Positional encodings
        self.src_pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.tgt_pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Encoder & Decoder stacks
        self.encoder = Encoder(num_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, n_heads, d_ff, dropout)

        # Final projection to Hindi vocabulary
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights with Xavier uniform
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # src: [batch_size, src_len]
        # returns: [batch_size, 1, 1, src_len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        # tgt: [batch_size, tgt_len]
        batch_size, tgt_len = tgt.size()
        # Padding mask: [batch_size, 1, 1, tgt_len]
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # Look-ahead causal mask: [1, 1, tgt_len, tgt_len]
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool().unsqueeze(0).unsqueeze(1)
        tgt_mask = tgt_pad_mask & causal_mask
        return tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Source embedding + scaling + pos encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.src_pos_encoding(src_emb)
        memory = self.encoder(src_emb, src_mask)

        # Target embedding + scaling + pos encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)
        out = self.decoder(tgt_emb, memory, tgt_mask, src_mask)

        # Output projection
        logits = self.fc_out(out)
        return logits

    @torch.no_grad()
    def translate(self, src_ids: list, tgt_tokenizer, max_len: int = 40, device: str = "cpu") -> str:
        self.eval()
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = self.make_src_mask(src_tensor)

        src_emb = self.src_embedding(src_tensor) * math.sqrt(self.d_model)
        src_emb = self.src_pos_encoding(src_emb)
        memory = self.encoder(src_emb, src_mask)

        ys = torch.tensor([[tgt_tokenizer.sos_idx]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(ys)
            tgt_emb = self.tgt_embedding(ys) * math.sqrt(self.d_model)
            tgt_emb = self.tgt_pos_encoding(tgt_emb)
            out = self.decoder(tgt_emb, memory, tgt_mask, src_mask)
            prob = self.fc_out(out[:, -1, :])
            next_word = torch.argmax(prob, dim=-1).item()

            ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
            if next_word == tgt_tokenizer.eos_idx:
                break

        output_tokens = ys.squeeze(0).tolist()
        return tgt_tokenizer.decode(output_tokens)
