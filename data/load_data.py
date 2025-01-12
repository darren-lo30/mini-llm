import os
import numpy as np
import torch

class TextDataloder():
  def __init__(self, path, split, block_size, batch_size, device):
    self.batch_size = batch_size
    self.block_size = block_size
    self.device = device
    if split == 'train':
      self.path = os.path.join(path, 'train.bin')
    else:
      self.path = os.path.join(path, 'val.bin')

  def __iter__(self):
    return self
  
  def __next__(self):
    data = np.memmap(self.path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
    if 'cuda' in self.device:
      # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
      x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
    else:
      x, y = x.to(self.device), y.to(self.device)
    return x, y