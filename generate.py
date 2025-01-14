from contextlib import nullcontext
from dataclasses import dataclass
import flash_attention
import torch
from omegaconf import MISSING
import hydra
from hydra.core.config_store import ConfigStore
import pickle
from gpt import GPT
import tiktoken
import os
import time

@dataclass
class GenerateConfig():
  checkpoint_path: str = MISSING
  data_path: str = MISSING
  device: str = 'cuda'
  attention_impl: str | None = None
  use_kv_cache: bool = False

def generate(config: GenerateConfig):
  checkpoint = torch.load(config.checkpoint_path)
  model_config = checkpoint['model_config']
  if config.attention_impl is not None:
    model_config.attention_impl = config.attention_impl
  model = GPT(model_config)
  model.load_state_dict(checkpoint['model'])

  model.eval()
  model.to(config.device)
  model = torch.compile(model)

  meta_path = os.path.join(config.data_path, 'meta.pkl')
  if os.path.exists(meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
  else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

  num_samples = 20
  num_new_tokens = 256
  temperature = 1.0
  start = '\n'
  start_ids = encode(start)
  x = (torch.tensor(start_ids, dtype=torch.long, device=config.device)[None, ...])
  dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not model_config.attention_impl == 'flash' else 'float16' 
  device_type = 'cuda' if 'cuda' in config.device else 'cpu' 
  ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
  ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

  t0 = time.time()
  with torch.no_grad():
    with ctx:
      for k in range(num_samples):
        y = model.generate(x, num_new_tokens, temperature=temperature, use_kv_cache=config.use_kv_cache)
        print(decode(y[0].tolist()))
        print('---------------')
  t1 = time.time()

  total_tokens = num_samples * num_new_tokens
  dt_ms = (t1 - t0) * 1000

  print(f'Generated {total_tokens} tokens in {dt_ms} ms. Average of {dt_ms / total_tokens} ms per token')

cs = ConfigStore.instance()
cs.store(name="base", node=GenerateConfig)

@hydra.main(version_base=None, config_path="./configs")
def main(cfg: GenerateConfig):
  generate(cfg)

if __name__ == '__main__':
  main()