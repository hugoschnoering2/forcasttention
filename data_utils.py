
import numpy as np
import pandas as pd

import torch

def create_dataset(file, train_per, seq_length, step_size):
  df = pd.read_csv(file)
  arr = df["open"].values
  returns = np.diff(arr) / arr[:-1]
  train_size = int(len(returns) * train_per)
  train, val = [], []
  for i in range(train_size):
    sample = returns[i: i + seq_length * step_size].astype(np.float32)
    train.append(np.split(sample, seq_length))
  for i in range(train_size, len(arr) - seq_length * step_size):
    sample = returns[i: i + seq_length * step_size].astype(np.float32)
    val.append(np.split(sample, seq_length))
  train, val = np.array(train), np.array(val)
  return TensorDataset(torch.from_numpy(train)), TensorDataset(torch.from_numpy(val))
