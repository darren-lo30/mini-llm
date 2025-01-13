from contextlib import nullcontext
from dataclasses import dataclass
from data.load_data import TextDataloder
from gpt import GPT, GPTConfig
import torch
import os
import hydra
from hydra.core.config_store import ConfigStore
import math 

@dataclass
class TrainConfig():
  model_config: GPTConfig 
  checkpoint_file: str | None = None
  batch_size: int = 8
  num_grad_acc_steps: int = 4
  device: str = 'cuda'
  data_path: str = './data'
  learning_rate: float = 6e-4
  num_steps: int = 10000

  # Training loop freqs
  eval_freq: int = 100
  log_freq: int = 100
  save_freq: int = 1000
  eval_iters: int = 50

  # Lr schedulers
  lr_warmup: int = 2000
  lr_decay: int = 10000
  min_learning_rate: float = 6e-5
  grad_clip: float | None = 1.0

  # Save options
  save_dir: str = './out'

def get_lr(config, step):
  assert config.lr_decay > config.lr_warmup
  assert config.min_learning_rate < config.learning_rate

  if step < config.lr_warmup:
    return step / config.lr_warmup

  if step <= config.lr_decay:
    ratio = (step - config.lr_warmup) / (config.lr_decay - config.lr_warmup)
    assert 0 <= ratio <= 1
    return config.min_learning_rate + math.cos(math.pi/2 * ratio) * (config.learning_rate - config.min_learning_rate)

  return config.min_learning_rate



def train(config: TrainConfig):  
  # Set up folders
  os.makedirs(config.save_dir, exist_ok=True)

  train_dataloader = TextDataloder(config.data_path, 'train', config.model_config.block_size, config.batch_size, config.device) 
  val_dataloader = TextDataloder(config.data_path, 'val', config.model_config.block_size, config.batch_size, config.device)

  train_iter = iter(train_dataloader)
  val_iter = iter(val_dataloader)

  # Precision code from NanoGPT
  torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
  torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
  device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
  # note: float16 data type will automatically use a GradScaler
  dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
  ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
  ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
  scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

  checkpoint = None
  if config.checkpoint_file is not None:
    checkpoint = torch.load(config.checkpoint_file)
    model = GPT(checkpoint['model_config'])
  else:
    model = GPT(config.model_config)

  model = model.to(device=config.device)
  model = torch.compile(model)
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=[0.9,0.999], eps=1e-8, weight_decay=1e-1)

  if checkpoint:
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


  step = 1 if not checkpoint else checkpoint['step']

  print('Starting Training')
  # Training loop
  while step <= config.num_steps:
    lr = get_lr(config, step)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr

    optimizer.zero_grad(set_to_none=True)    
    model.train()
    
    average_train_loss = 0
    for acc_step in range(config.num_grad_acc_steps):
      inputs, targets = next(train_iter)
      assert inputs.device == targets.device
      
      with ctx:
        logits = model(inputs)
      loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
      loss = loss / config.num_grad_acc_steps
      average_train_loss += loss.item()

      scaler.scale(loss).backward()  

    if config.grad_clip is not None:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    scaler.step(optimizer)
    scaler.update()
      
    if step % config.log_freq == 0:
      print(f"Step {step}/{config.num_steps}, Train Loss: {average_train_loss:.4f}")

    # Model validation
    if step % config.eval_freq == 0:
      model.eval()
      with torch.no_grad():
        val_loss = 0
        for _ in range(config.eval_iters):
          inputs, targets = next(val_iter)
          with ctx:
            logits = model(inputs)
          val_loss += torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)                
        val_loss = val_loss.item() / config.eval_iters
      print(f"Validation Loss: {val_loss:.4f}")

    if step % config.save_freq == 0:
      checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'model_config': config.model_config,
      }
      print(f"Saving checkpoint to {config.save_dir}")
      torch.save(checkpoint, os.path.join(config.save_dir, f'ckpt_{step}.pt'))
    
    step += 1

  print("Training complete.")

cs = ConfigStore.instance()
cs.store(name="base", node=TrainConfig)

@hydra.main(version_base=None, config_path="./configs")
def main(cfg: TrainConfig):
  train(cfg)

if __name__ == '__main__':
  main()