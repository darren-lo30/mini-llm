from dataclasses import dataclass
from logging import config
from torch import nn as nn
import torch
import math
import inspect
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
  def __init__(self, config):
    super().__init__()
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

  def forward(self, Q, K, V, is_causal, softmax_scale):
    seq_len = Q.shape[2]
    key_len = K.shape[2]

    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if is_causal:
      P = P.masked_fill(torch.ones_like(self.bias[:,:,:seq_len,:key_len]) == 0, float('-inf'))
    P = torch.nn.functional.softmax(P, dim=-1)
    out = torch.matmul(P, V)

    return out
  

class FFN(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    hidden_dim_size = config.ffn_hidden_dim_factor * config.embed_size
    self.lin1 = nn.Linear(config.embed_size, hidden_dim_size)
    self.gelu = nn.GELU()
    self.lin2 = nn.Linear(hidden_dim_size, config.embed_size)
    self.dropout = nn.Dropout(config.p_dropout)

  def forward(self, x):
    x = self.lin1(x)
    x = self.gelu(x)
    x = self.lin2(x)
    x = self.dropout(x)

    return x
  
class MultiHeadAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.Q_proj = nn.Linear(config.embed_size, config.embed_size) 
    self.K_proj = nn.Linear(config.embed_size, config.embed_size) 
    self.V_proj = nn.Linear(config.embed_size, config.embed_size)

    self.config = config
    self.head_embed_size = config.embed_size // config.num_attn_heads
    self.softmax_scale = 1.0 / math.sqrt(self.head_embed_size)

    if config.attention_impl == 'flash':
      from flash_attention import FlashAttentionFn
      self.attention = FlashAttentionFn.apply
    else:
      self.attention = Attention(config)

    self.lin_out = nn.Linear(config.embed_size, config.embed_size)

  def forward(self, QKV, is_causal, kv_cache = None): 
    batch_size, seq_len = QKV.shape[0], QKV.shape[1]

    if kv_cache is not None:
      assert self.config.attention_impl != 'flash'
    
    Q = self.Q_proj(QKV)
    K = self.K_proj(QKV)
    V = self.V_proj(QKV) 
    # [batch_size, seq_len, embed_dim]
    updated_kv_cache = kv_cache
    if kv_cache:
      K_cache, V_cache = kv_cache
      K = K if K_cache is None else torch.cat([K_cache, K], dim=1)
      V = V if V_cache is None else torch.cat([V_cache, V], dim=1)
      
      # Truncate KV cache
      if K.shape[1] >= self.config.block_size:
        K = K[:, -self.config.block_size:, :]
        V = V[:, -self.config.block_size:, :]
      updated_kv_cache = (K, V)

    # QKV <- [batch_size, seq_len, embed_dim]
    # Projected <- [batch_size, seq_len, embed_dim]
    Q = Q.view(batch_size, -1, self.config.num_attn_heads, self.head_embed_size).transpose(1, 2).contiguous()
    K = K.view(batch_size, -1, self.config.num_attn_heads, self.head_embed_size).transpose(1, 2).contiguous()
    V = V.view(batch_size, -1, self.config.num_attn_heads, self.head_embed_size).transpose(1, 2).contiguous()
    # Q, K, V <- [batch_size, num_heads, seq_len, embed_dim // num_heads]

    # if self.config.attention_impl == 'flash':
    #   out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.config.p_dropout if self.training else 0, is_causal=True)
    # else:

    out = self.attention(Q, K, V, is_causal, self.softmax_scale) # Same shape as Q, reshape
    # out: [batch_size, num_heads, seq_len, embed_dim // num_heads]
    assert out.shape == (Q.shape[0], self.config.num_attn_heads, seq_len, self.head_embed_size)
    out = out.transpose(1, 2).contiguous()
    out = out.view(batch_size, -1, self.config.embed_size)
    out = self.lin_out(out)
    return out, updated_kv_cache
  
class TransformerBlock(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.embed_size, bias=config.bias)
    self.mha = MultiHeadAttention(config)
    self.ln_2 = nn.LayerNorm(config.embed_size, bias=config.bias)
    self.ffn = FFN(config)
    self.dropout = nn.Dropout(config.p_dropout)

  def forward(self, x, kv_cache = None):
    y, kv_cache = self.mha(self.ln_1(x), is_causal=True, kv_cache=kv_cache)
    x = x + y
    x = x + self.dropout(self.ffn(self.ln_2(x)))
    return x, kv_cache


class GPT(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config

    self.out_linear = nn.Linear(config.embed_size, config.vocab_size)
    self.layer_norm = nn.LayerNorm(config.embed_size, bias=config.bias)

    self.token_embed = nn.Embedding(config.vocab_size, config.embed_size)
    self.pos_embed = nn.Embedding(config.block_size, config.embed_size)
  
    self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

    self.apply(self._init_weights)

  def get_pos_embeds(self, num, device):
    indices = torch.arange(0, num, device=device)
    return self.pos_embed(indices)
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, text, kv_cache = None):
    use_kv_cache = kv_cache is not None
    device = text.device

    batch_size, seq_len = text.shape
    assert seq_len <= self.config.block_size

    pos_embeds = self.get_pos_embeds(seq_len, device=device)
    token_embeds = self.token_embed(text)

    embeddings = pos_embeds + token_embeds # [batch_size, seq_len, embed_size]

    if not kv_cache:
      kv_cache = [None] * self.config.num_layers
    else:
      # kv_cache self.config.num_layers * ((k, v)) of respective layers
      assert len(kv_cache) == self.config.num_layers
      # Only process most recent value
      num_kv_cached = 0 if kv_cache[0][0] is None else kv_cache[0][0].shape[1]
      update_idx = -1 if num_kv_cached >= self.config.block_size else num_kv_cached
      embeddings = embeddings[:, [update_idx], :]

    updated_kv_cache = [] 
    out = embeddings
    for transformer_block, block_kv_cache in zip(self.transformer_blocks, kv_cache):
      out, updated_block_kv_cache = transformer_block(out, block_kv_cache)
      updated_kv_cache.append(updated_block_kv_cache)

    out = self.layer_norm(out)
    out = self.out_linear(out)

    # Return logits

    return out, updated_kv_cache if use_kv_cache else None

  @torch.no_grad()
  def generate(self, idx, num_tokens, temperature=1.0, use_kv_cache=False):
    # Flash Attention Impl requires padded seq length
    kv_cache = [(None, None) for _ in range(self.config.num_layers)] if use_kv_cache else None 
    assert idx.shape == (1, 1)
    padding = self.config.block_size
    idx = torch.nn.functional.pad(idx, (0, self.config.block_size - 1), mode='constant', value=0)

    for i in range(1, num_tokens + 1):
      idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
      logits, kv_cache = self.forward(idx_cond, kv_cache=kv_cache)
      # print(logits)
      if use_kv_cache:
        assert logits.size(1) == 1
        logits = logits[:, -1, :] / temperature
      elif i < self.config.block_size:
        logits = logits[:, i - 1, :] / temperature
      else:
        logits = logits[:, -1, :] / temperature

      probs = torch.nn.functional.softmax(logits, dim=-1)
      # print(probs)
      idx_next = torch.multinomial(probs, num_samples=1)

      if i < self.config.block_size:
        idx[0, i] = idx_next
      else:
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
