import sys
sys.path.append(".")

import os

import yaml
import datetime


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from data_utils import create_dataset_autoencoding as create_dataset
from utils import _parse_args

from sporco.dictlrn import bpdndl
from sporco.admm import bpdn

if __name__ == "__main__":

    config, config_yaml = _parse_args()

    if config.seed is not None:
        np.random.seed(config.seed)

    results_folder = os.path.join(config.results_folder,
    "_".join([datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), config.model]))
    os.mkdir(results_folder)
    print("Results will be saved here : " + str(results_folder))

    with open(os.path.join(results_folder, "config.yaml"), "w") as f:
        f.write(yaml.dump(config_yaml))

    train_dataset, val_dataset = create_dataset(config.file, config.train_per, 1, config.sample_size,
                                                config.data_transform, config.standardization, tensor=False)

    D0 = np.random.randn(train_dataset.shape[2], config.embed_size)

    opt = bpdndl.BPDNDictLearn.Options({'Verbose': False, 'MaxMainIter': 100,
                                        'BPDN': {'rho': 10.0*config.lambda_l + 0.1},
                                         'CMOD': {'rho': train_dataset.shape[0] / 1e3}})
    d = bpdndl.BPDNDictLearn(D0, train_dataset.squeeze(1).T, config.lambda_l, opt)
    d.solve()
    D1 = d.getdict()

    plt.figure(figsize=(10, 5*config.embed_size))
    for i in range(config.embed_size):
        plt.subplot(config.embed_size, 1, i+1)
        plt.plot(D1[:, i], label="basis function n".format(i))
        plt.legend()
    plt.savefig(os.path.join(results_folder, "basis_functions.jpg"))
    plt.close('all')

    r = bpdn.BPDN(D1, val_dataset.squeeze(1).T, config.lambda_r)
    reconstruction = (D1 @ r.solve()).T

    plt.figure(figsize=(10, 5*32))
    for i in range(32):
        plt.subplot(32, 1, i+1)
        j = np.random.choice(len(val_dataset))
        plt.plot(val_dataset[j, 0, :], label="true")
        plt.plot(reconstruction[j, :], label="pred")
        plt.legend()
    plt.savefig(os.path.join(results_folder, "reconstruction.jpg"))
    plt.close('all')

    mse = np.linalg.norm((val_dataset.squeeze(1)-reconstruction), ord=2)**2 / len(val_dataset)

    results = pd.DataFrame({"mse_loss" : [mse]})
    results.to_csv(os.path.join(results_folder, "results.csv"))
