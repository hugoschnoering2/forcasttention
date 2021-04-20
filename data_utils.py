
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset

def create_dataset(file, train_per, seq_length, step_size, transformation):
  df = pd.read_csv(file)
  data = df["open"].values
  if transformation == "rdiff":
      data =  np.diff(data) / data[:-1]
  elif transformation == "logr":
      data = np.log(data[1:] / data[:-1])
  else:
      pass
  train_size = int(len(data) * train_per)
  train, val = [], []
  for i in range(train_size):
    sample = data[i: i + seq_length * step_size].astype(np.float32)
    train.append(np.split(sample, seq_length))
  for i in range(train_size, len(arr) - seq_length * step_size):
    sample = data[i: i + seq_length * step_size].astype(np.float32)
    val.append(np.split(sample, seq_length))
  train, val = np.array(train), np.array(val)
  return TensorDataset(torch.from_numpy(train)), TensorDataset(torch.from_numpy(val))
