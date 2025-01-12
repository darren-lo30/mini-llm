from dataclasses import dataclass
from torch import nn as nn
import torch
import math

@dataclass
class GPTConfig:
  embed_size: int
  vocab_size: int
  block_size: int
  num_layers: int 
  ffn_hidden_dim_factor: int
  p_dropout: float
  num_attn_heads: int
  attention_impl: str
  bias: bool

class Attention(nn.Module):
  def forward(self, Q, K, V, is_causal, softmax_scale):
    seq_len = Q.shape[2]

    mask = torch.tril(torch.ones((seq_len, seq_len), device=Q.device))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if is_causal:
      P[:, :, mask == 0] = float("-inf")
    P = torch.nn.functional.softmax(P.float(), dim=-1)
    out = torch.matmul(P, V)

    return out
  

class FFN(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    hidden_dim_size = config.ffn_hidden_dim_factor * config.embed_size
    self.lin1 = nn.Linear(config.embed_size, hidden_dim_size)
    self.gelu = nn.GELU()
    self.lin2 = nn.Linear(hidden_dim_size, config.embed_size)

  def forward(self, x):
    x = self.lin1(x)
    x = self.gelu(x)
    x = self.lin2(x)

    return x
  
class MultiHeadAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.Q_proj = nn.Linear(config.embed_size, config.embed_size) 
    self.K_proj = nn.Linear(config.embed_size, config.embed_size) 
    self.V_proj = nn.Linear(config.embed_size, config.embed_size)

    self.config = config
    self.head_embed_size = config.embed_size // config.num_attn_heads
    self.softmax_scale = math.sqrt(self.head_embed_size)

    if config.attention_impl == 'flash':
      from flash_attention import FlashAttention
      self.attention = FlashAttention()
    else:
      self.attention = Attention()

    self.lin_out = nn.Linear(config.embed_size, config.embed_size)

  def forward(self, QKV, is_causal): 
    batch_size = QKV.shape[0]

    Q = self.Q_proj(QKV).view(batch_size, -1, self.config.num_attn_heads, self.head_embed_size)
    K = self.K_proj(QKV).view(batch_size, -1, self.config.num_attn_heads, self.head_embed_size)
    V = self.V_proj(QKV).view(batch_size, -1, self.config.num_attn_heads, self.head_embed_size)

    out = self.attention(Q, K, V, is_causal = is_causal, softmax_scale = self.softmax_scale) # Same shape as Q, reshape
    out = out.view(batch_size, -1, self.config.embed_size)
    out = self.lin_out(out)
    return out
  
class TransformerBlock(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.embed_size, bias=config.bias)
    self.mha = MultiHeadAttention(config)
    self.ln_2 = nn.LayerNorm(config.embed_size, bias=config.bias)
    self.ffn = FFN(config)
    self.dropout = nn.Dropout(config.p_dropout)

  def forward(self, x):
    x = x + self.mha(self.ln_1(x), is_causal=True)
    x = x + self.dropout(self.ffn(self.ln_2(x)))
    return x


class GPT(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config

    self.out_linear = nn.LazyLinear(config.vocab_size)
    self.softmax = nn.Softmax()
    self.layer_norm = nn.LayerNorm(config.embed_size, bias=config.bias)

    self.token_embed = nn.Embedding(config.vocab_size, config.embed_size)
    self.pos_embed = nn.Embedding(config.block_size, config.embed_size)
  
    self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

  def get_pos_embeds(self, num, device):
    indices = torch.arange(0, num, device=device)
    return self.pos_embed(indices)
  
  def forward(self, text, kv_cache = None):
    device = text.device

    batch_size, seq_len = text.shape
    
    pos_embeds = self.get_pos_embeds(seq_len, device=device)
    token_embeds = self.token_embed(text)

    embeddings = pos_embeds + token_embeds # [batch_size, seq_len, embed_size]

    out = embeddings
    for transformer_block in self.transformer_blocks:
      out = transformer_block(out)

    out = self.layer_norm(out)
    out = self.out_linear(out)

    # Return logits

    return out
  
  @torch.no_grad()
  def generate(self, idx, num_tokens, temperature=1.0):
    for i in range(num_tokens):
      idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
      logits = self.forward(idx_cond)
      logits = logits[:, -1, :] / temperature
      probs = torch.nn.functional.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx







    

