from contextlib import nullcontext
import torch

from gpt import GPT
import tiktoken

ckpt_path = './asd'
device = 'cuda'

def generate():
  assert ckpt_path is not None

  checkpoint = torch.load(ckpt_path)
  model = GPT(checkpoint['model_config'])
  model.load_state_dict(checkpoint['model'])

  model.eval()
  model.to(device)

  enc = tiktoken.get_encoding("gpt2")
  encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
  decode = lambda l: enc.decode(l)
  num_samples = 20
  num_new_tokens = 100
  temperature = 1.0
  start = '\n'
  start_ids = encode(start)
  x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
  dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
  device_type = 'cuda' if 'cuda' in device else 'cpu' 
  ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
  ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

  with torch.no_grad():
    with ctx:
      for k in range(num_samples):
        y = model.generate(x, num_new_tokens, temperature=temperature)
        print(decode(y[0].tolist()))
        print('---------------')


  
if __name__ == '__main__':
  generate()