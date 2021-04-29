
import sys
sys.path.append(".")

import os

import yaml
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import create_dataset_autoencoding as create_dataset
from utils import _parse_args
from soft_dtw_loss import SoftDTW

from earlyStopping import EarlyStopping

def train_one_epoch(dataloader, model, loss, optimizer, device):
  running_loss = 0.
  for batch in dataloader:
    model.train()
    model.zero_grad()
    input = batch[0].to(device).permute(1, 0, 2)
    batch_size = input.shape[1]
    pred = model(input)
    pred_error = loss(input.permute(1, 0, 2).reshape(batch_size, -1),
                      pred.permute(1, 0, 2).reshape(batch_size, -1))
    running_loss += pred_error.item()
    pred_error.backward()
    optimizer.step()
  return running_loss / len(dataloader)

def evaluate_one_epoch(dataloader, model, loss, device):
  running_loss = 0.
  sdtw = SoftDTW()
  running_dtw_loss = 0.
  with torch.no_grad():
      for batch in dataloader:
        model.eval()
        input = batch[0].to(device).permute(1, 0, 2)
        batch_size = input.shape[1]
        pred = model(input)
        pred_error = loss(input.permute(1, 0, 2).reshape(batch_size, -1),
                          pred.permute(1, 0, 2).reshape(batch_size, -1))
        running_loss += pred_error.item()
        running_dtw_loss += sdtw(input.permute(1, 0, 2).reshape(batch_size, -1),
                                    pred.permute(1, 0, 2).reshape(batch_size, -1)).item()
  return running_loss / len(dataloader), running_dtw_loss / len(dataloader)

def plot_pred(batch, model, device, path):
    with torch.no_grad():
        model.eval()
        input = batch[0].to(device).permute(1, 0, 2)
        pred = model(input)
        seq_length, batch_size, step_size = input.shape
        plt.figure(figsize=(10, 5*batch_size))
        for i in range(batch_size):
            plt.subplot(batch_size, 1, i+1)
            plt.plot(input[:, i, :].cpu().numpy().ravel(), label="true")
            plt.plot(pred[:, i, :].detach().cpu().numpy().ravel(), label="pred")
            plt.legend()
        plt.savefig(path)
        plt.close('all')

if __name__ == "__main__":

    config, config_yaml = _parse_args()

    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    results_folder = os.path.join(config.results_folder,
    "_".join([datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), config.model]))
    os.mkdir(results_folder)
    print("Results will be saved here : " + str(results_folder))

    with open(os.path.join(results_folder, "config.yaml"), "w") as f:
        f.write(yaml.dump(config_yaml))

    train_dataset, val_dataset = create_dataset(config.file, config.train_per, 1, config.step_size,
                                                config.data_transform, config.standardization)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from models.rnn.lstm import CNN_autoencoder
    model = CNN_autoencoder(config.step_size, config.embed_size).to(device)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, cooldown=20, verbose=True)
    es_encoder = EarlyStopping(path=os.path.join(results_folder, "cnn_encoder.pt"), patience=30)
    es_decoder = EarlyStopping(path=os.path.join(results_folder, "cnn_decoder.pt"), patience=30)

    for plot_batch in train_dataloader:
        break

    train_loss = []
    val_loss = []
    dtw_loss = []

    for epoch in range(config.num_epochs):
        tl = train_one_epoch(train_dataloader, model, loss, optimizer, device)
        vl, dtwl = evaluate_one_epoch(val_dataloader, model, loss, device)
        stop = es_encoder(vl, model.encoder)
        es_decoder(vl, model.decoder)
        scheduler.step(vl)
        train_loss.append(tl)
        val_loss.append(vl)
        dtw_loss.append(dtwl)
        print(" EPOCH {0} #### TRAIN LOSS {1} #### VAL LOSS {2} ".format(epoch, train_loss[-1], val_loss[-1]))
        if epoch % config.plot_step == 0:
            path = os.path.join(results_folder, "epoch " + str(epoch) + ".jpg")
            plot_pred(plot_batch, model, device, path)
        if stop:
            break

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.yscale('log')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(dtw_loss, label="val soft dtw loss")
    plt.legend()
    plt.savefig(os.path.join(results_folder, "train_monitor.jpg"))
    plt.close("all")

    df = pd.DataFrame({"train_loss" : train_loss, "val_loss" : val_loss, "val_dtw_loss" : dtw_loss})
    df.to_csv(os.path.join(results_folder, "train_monitor.csv"))
