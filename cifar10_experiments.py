'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import numpy as np

from lib.densenet import DenseNet121
from lib.experiments import load_dataset, run_wcl, train_with_original, run_pop, run_egdis, run_cl

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--experiment', default=1, type=int, help='Run which experiment')
parser.add_argument('--select', default=1, type=int, help='Run which stage')
parser.add_argument('--batch_size', default=256, type=int, help='Traning batch size')
parser.add_argument('--stage', default=1, type=int, help='Run which substage')
parser.add_argument('--numbers', default=0, type=int, help='Run which fixed number of samples')
args = parser.parse_args()

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("cifar10")
print("cifar10 loaded")

print("There are {} training samples and {} validation samples".format(x_train.shape[0], x_valid.shape[0]))
print("There are {} test samples.".format(x_test.shape[0]))

batch_size = args.batch_size
net = DenseNet121()

print("This is records for stage {}".format(args.stage))

numbers = args.numbers

# def load_net(weights):
#     new_state_dict = OrderedDict()
#     for k, v in weights.items():
#         name = k[7:]  # remove module.
#         new_state_dict[name] = v
#     return new_state_dict


# Experiment 1: train whole cifar10  with DenseNet121
if args.experiment == 1:
    print("Train with the whole dataset.")
    history = train_with_original((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10",
                                  batch_size=batch_size, stage=args.stage)

    np.save(os.path.join(os.getcwd(), "models", "cifar10", "whole_train_his" + "_stage_" + str(args.stage) + ".npy"),
            history)
    print("History saved.")

# Experiment 2: train the POP selected dataset
if args.experiment == 2:
    print("Train with the pop selected dataset.")
    history = run_pop((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10,
                      batch_size=batch_size, i=args.select, stage=args.stage, num_samples=numbers)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar10",
                             "pop_his_size_" + str(his["size"]) + "_stage_" + str(args.stage) + ".npy"), history)
    print("History saved.")

# Experiment 3: train the EGDIS selected dataset
if args.experiment == 3:
    print("Train with the egdis selected dataset.")
    history = run_egdis((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10,
                        batch_size=batch_size, stage=args.stage)
    np.save(os.path.join(os.getcwd(), "models", "cifar10", "egdis_his" + "_stage_" + str(args.stage) + ".npy"), history)
    print("History saved.")

# Experiment 4: train the CL selected dataset
if args.experiment == 4:
    print("Train with the cl selected dataset.")
    history = run_cl((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10,
                     batch_size=batch_size, i=args.select, stage=args.stage, num_samples=numbers)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar10",
                             "cl_his_size_" + str(his["size"]) + "_stage_" + str(args.stage) + ".npy"), history)
    print("History saved.")

# Experiment 5: train the WCL selected dataset
if args.experiment == 5:
    print("Train with the im wcl selected dataset.")
    history = run_wcl((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10,
                      batch_size=batch_size, i=args.select, stage=args.stage, num_samples=numbers)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar10",
                             "im_wcl_his_size_" + str(his["size"]) + "_stage_" + str(args.stage) + ".npy"), history)
    print("History saved.")

# if args.experiment == 6:
#     print("Train with the wcl selected dataset (by weighted the boundary points).")
#     history = run_wcl2((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10,
#                        batch_size=batch_size, i=args.select, stage=args.stage, num_samples=numbers)
#     for his in history:
#         np.save(os.path.join(os.getcwd(), "models", "cifar10",
#                              "wcl2_his_size_" + str(his["size"]) + "_stage_" + str(args.stage) + ".npy"), history)
#     print("History saved.")
#
# if args.experiment == 7:
#     print("Train with the wcl selected dataset (by weighted the boundary points).")
#     history = run_wcl3((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10,
#                        batch_size=batch_size, i=args.select, stage=args.stage, num_samples=numbers)
#     for his in history:
#         np.save(os.path.join(os.getcwd(), "models", "cifar10",
#                              "wcl3_his_size_" + str(his["size"]) + "_stage_" + str(args.stage) + ".npy"), history)
#     print("History saved.")
