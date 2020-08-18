'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import numpy as np

from lib.densenet import DenseNet121
from lib.experiments import load_dataset, collect_wcl

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("cifar10")
print("cifar10 loaded")

print("There are {} training samples and {} validation samples".format(x_train.shape[0], x_valid.shape[0]))
print("There are {} test samples.".format(x_test.shape[0]))

# x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("cifar100")
# print("cifar100 loaded")
# print("There are {} training samples and {} validation samples".format(x_train.shape[0], x_valid.shape[0]))
# print("There are {} test samples.".format(x_test.shape[0]))


# batch_size = args.batch_size
# net = DenseNet121(100)
batch_size = 256
#
# Experiment 5: train the WCL selected dataset

print("Train with the im wcl selected dataset.")
for i in range(800, 850, 5):
    history = collect_wcl((x_train, y_train), (x_valid, y_valid), (x_test, y_test), None, "cifar10", 10,
                  batch_size=batch_size, i=i)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar10", "framework",
                         "im_wcl_his_size_" + str(his["size"]) + ".npy"), history)
print("History saved.")