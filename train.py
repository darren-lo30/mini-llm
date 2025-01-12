from dataclasses import dataclass
from data.load_data import TextDataloder
from gpt import GPT, GPTConfig
import torch
import os
import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class TrainConfig():
  model_config: GPTConfig 
  checkpoint_file: str | None = None
  batch_size: int = 8
  num_grad_acc_steps: int = 4
  device: str = 'cuda'
  data_path: str = './data'
  learning_rate: float = 0.001
  num_steps: int = 10000

  # Training loop freqs
  eval_freq: int = 100
  log_freq: int = 100
  save_freq: int = 1000

  # Save options
  save_dir: str = './out'

def train(config: TrainConfig):  
  # Set up folders
  os.makedirs(config.save_dir, exist_ok=True)

  train_dataloader = TextDataloder(config.data_path, 'train', config.model_config.block_size, config.batch_size, config.device) 
  val_dataloader = TextDataloder(config.data_path, 'val', config.model_config.block_size, config.batch_size, config.device)

  train_iter = iter(train_dataloader)
  val_iter = iter(val_dataloader)

  scaler = torch.amp.GradScaler()

  checkpoint = None
  if config.checkpoint_file is not None:
    checkpoint = torch.load(config.checkpoint_file)
    model = GPT(checkpoint['model_config'])
  else:
    model = GPT(config.model_config)

  model = model.to(device=config.device)
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=[0.9,0.999], eps=1e-8)

  if checkpoint:
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


  step = 1 if not checkpoint else checkpoint['step']
  running_loss = 0.0

  print('Starting Training')
  # Training loop
  while step <= config.num_steps:
    optimizer.zero_grad(set_to_none=True)    
    model.train()
    
    for acc_step in range(config.num_grad_acc_steps):
      inputs, targets = next(train_iter)
      assert inputs.device == targets.device
      
      
      logits = model(inputs)
      loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
      loss = loss / config.num_grad_acc_steps
      running_loss += loss.item()

      scaler.scale(loss).backward()      
    scaler.step(optimizer)
    scaler.update()
      
    if step % config.log_freq == 0:
      print(f"Step {step}/{config.num_steps}")
      print(f"Batch {step}, Loss: {loss.item():.4f}")

      avg_train_loss = running_loss / step
      print(f"Average Training Loss: {avg_train_loss:.4f}")

    # Model validation
    if step % config.eval_freq == 0:
      model.eval()
      with torch.no_grad():
        inputs, targets = next(val_iter)
        logits = model(inputs)
        val_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)                
      print(f"Validation Loss: {val_loss:.4f}")

    if step % config.save_freq == 0:
      checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'model_config': config.model_config,
      }
      print(f"Saving checkpoint to {config.save_dir}")
      torch.save(checkpoint, os.path.join(config.save_dir, 'ckpt.pt'))
    
    step += 1

  print("Training complete.")

cs = ConfigStore.instance()
cs.store(name="base", node=TrainConfig)

@hydra.main(version_base=None, config_path="./configs")
def main(cfg: TrainConfig):
  train(cfg)

if __name__ == '__main__':
  main()