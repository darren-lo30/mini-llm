from dataclasses import dataclass
from torch import nn as nn
import torch

from attention import FlashAttention

@dataclass
class GPTConfig:
  embed_size: int
  vocab_size: int
  block_size: int
  num_layers: int 
  ffn_hidden_dim: int
  p_dropout: float
  num_attn_heads: int

class FFN():
  def __init__(self, config: GPTConfig):
    self.lin1 = nn.Linear(config.embed_size, config.ffn_hidden_dim)
    self.gelu = nn.GELU()
    self.lin2 = nn.Linear(config.ffn_hidden_dim, config.embed_size)

  def forward(self, x):
    x = self.lin1(x)
    x = self.gelu(x)
    x = self.lin2(x)

    return x
  
class MultiHeadAttention():
  def __init__(self, config: GPTConfig):
    self.Q_proj = nn.Linear(config.embed_size, config.embed_size) 
    self.K_proj = nn.Linear(config.embed_size, config.embed_size) 
    self.V_proj = nn.Linear(config.embed_size, config.embed_size)

    self.config = config
    self.head_embed_size = config.embed_size // config.num_attn_heads
    self.softmax_scale = torch.sqrt(self.head_embed_size)

    self.attention = FlashAttention()
    self.lin_out = nn.Linear(config.embed_size, config.embed_size)

  def forward(self, Q, K, V, is_causal = False): 
    batch_size = Q.shape[0]

    Q = self.Q_proj(Q).view(batch_size, -1, self.config.num_attn_heads, self.head_embed_size)
    K = self.K_proj(K).view(batch_size, -1, self.config.num_attn_heads, self.head_embed_size)
    V = self.V_proj(V).view(batch_size, -1, self.config.num_attn_heads, self.head_embed_size)

    out = FlashAttention(Q, K, V, is_causal = is_causal, softmax_scale = self.softmax_scale) # Same shape as Q, reshape
    out = out.view(batch_size, -1, self.config.embed_size)
    out = self.lin_out(out)
    return out
  
class TransformerBlock():
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
    self.mha = MultiHeadAttention()
    self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
    self.ffn = nn.FFN(config)
    self.dropout = nn.Dropout(config.p_dropout)

  def forward(self, x):
    x = x + self.mha(self.ln_1(x))
    x = x + self.dropout(self.ffn(self.ln_2(x)))
    return x


class GPT():
  def __init__(self, config: GPTConfig):
    self.config = config

    self.out_linear = nn.LazyLinear(config.vocab_size)
    self.softmax = nn.Softmax()
    self.layer_norm = nn.LayerNorm()

    self.token_embed = nn.Embedding(config.vocab_size, config.embed_size),
    self.pos_embed = nn.Embedding(config.block_size, config.embed_size),
  
    self.transformer_blocks = [TransformerBlock() for _ in range(config.num_layers)]

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
    out = self.softmax(out)

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







    

