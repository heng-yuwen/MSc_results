'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import numpy as np

from lib.densenet import DenseNet121
from lib.experiments import load_dataset, run_wcl, train_with_original, run_pop, run_egdis, run_cl, run_bwcl

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--experiment', default=1, type=int, help='Run which experiment')
parser.add_argument('--select', default=1, type=int, help='Run which stage')
parser.add_argument('--batch_size', default=256, type=int, help='Traning batch size')
parser.add_argument('--stage', default=1, type=int, help='Run which substage')
parser.add_argument('--numbers', default=0, type=int, help='Run which fixed number of samples')
parser.add_argument('--wcl', default=0, type=int, help='Use weighted rather than all boundary points')

args = parser.parse_args()

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("cifar100")
x_train_idx = np.load(os.path.join(os.getcwd(), "datasets", "cifar40", "train_idx.npy"))
x_valid_idx = np.load(os.path.join(os.getcwd(), "datasets", "cifar40", "valid_idx.npy"))
x_test_idx = np.load(os.path.join(os.getcwd(), "datasets", "cifar40", "test_idx.npy"))

print("cifar40 loaded")
print("There are {} training samples and {} validation samples".format(len(x_train_idx), len(x_valid_idx)))
print("There are {} test samples.".format(len(x_test_idx)))

x_train = x_train[x_train_idx]
y_train = y_train[x_train_idx]

x_valid = x_valid[x_valid_idx]
y_valid = y_valid[x_valid_idx]

x_test = x_test[x_test_idx]
y_test = y_test[x_test_idx]

unique_y = np.unique(y_train)
y_train = np.array([np.argwhere(unique_y == y)[0][0] for y in y_train])
y_valid = np.array([np.argwhere(unique_y == y)[0][0] for y in y_valid])
y_test = np.array([np.argwhere(unique_y == y)[0][0] for y in y_test])

batch_size = args.batch_size
net = DenseNet121(num_classes=40)

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
    history = train_with_original((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar40",
                                  batch_size=batch_size, stage=args.stage)

    np.save(os.path.join(os.getcwd(), "models", "cifar40", "whole_train_his" + "_stage_" + str(args.stage) + ".npy"),
            history)
    print("History saved.")

# Experiment 2: train the POP selected dataset
if args.experiment == 2:
    print("Train with the pop selected dataset.")
    history = run_pop((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar40", 40,
                      batch_size=batch_size, i=args.select, stage=args.stage, num_samples=numbers)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar40",
                             "pop_his_size_" + str(his["size"]) + "_stage_" + str(args.stage) + ".npy"), history)
    print("History saved.")

# Experiment 3: train the EGDIS selected dataset
if args.experiment == 3:
    print("Train with the egdis selected dataset.")
    history = run_egdis((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar40", 40,
                        batch_size=batch_size, stage=args.stage)
    np.save(os.path.join(os.getcwd(), "models", "cifar40", "egdis_his" + "_stage_" + str(args.stage) + ".npy"),
            history)
    print("History saved.")

# Experiment 4: train the CL selected dataset
if args.experiment == 4:
    print("Train with the cl selected dataset.")
    history = run_cl((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar40", 40,
                     batch_size=batch_size, i=args.select, stage=args.stage, num_samples=numbers)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar40",
                             "cl_his_size_" + str(his["size"]) + "_stage_" + str(args.stage) + ".npy"), history)
    print("History saved.")

# Experiment 5: train the WCL selected dataset
if args.experiment == 5:
    print("Train with the im wcl selected dataset.")
    history = run_wcl((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar40", 40,
                      batch_size=batch_size, i=args.select, stage=args.stage, num_samples=numbers)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar40",
                             "im_wcl_his_size_" + str(his["size"]) + "_stage_" + str(args.stage) + ".npy"), history)
    print("History saved.")

# Experiment 6: train the WCL selected dataset
if args.experiment == 6:
    print("Train with the bwcl selected dataset.")
    history = run_bwcl((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar40", 40,
                      batch_size=batch_size, i=args.select, stage=args.stage, num_samples=numbers)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar40",
                             "im_bwcl_his_size_" + str(his["size"]) + "_stage_" + str(args.stage) + ".npy"), history)
    print("History saved.")