# check gpu devices
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os
import random
import numpy as np
# load cifar10 datasets
from lib.data_loader import load_cifar100
from lib.feature_extractor import NASNetLargeExtractor

devices = tf.config.list_physical_devices('GPU')
if len(devices) < 1:
    raise AttributeError("No GPU found!")
else:
    print(devices)
    print()

batch_size = 128

# download google nasnet large pre-trained model
model = NASNetLargeExtractor(32, 100, model_path="models/cifar100", data_path="datasets/cifar100")
print("Pre-trained NASNetLarge is loaded.")

# preprocess the dataset
(x_train, y_train), (x_test, y_test) = load_cifar100()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


def preprocess_data(data_set):
    data_set /= 255.0
    return data_set


x_train = preprocess_data(x_train)
x_test = preprocess_data(x_test)

# select subsets
def subset(seed, size):
    random.seed(seed)
    classes = np.random.choice(range(100), size, replace=False)
    train_idx = np.array([]).astype("int64")
    valid_idx = np.array([]).astype("int64")
    test_idx = np.array([]).astype("int64")
    for class_id in classes:
        train_idx = np.union1d(train_idx, np.argwhere(y_train == class_id))
        test_idx = np.union1d(test_idx, np.argwhere(y_test == class_id))
        valid_idx = np.union1d(valid_idx, np.argwhere(y_valid == class_id))
        print(len(train_idx))
        print(len(test_idx))
        print(len(valid_idx))
    model.extracted_features = extracted_train_features[train_idx]

    # use dense layer to test feature quality
    history = model.train_classifier(y_train[train_idx], epochs=500, batch_size=batch_size, validation_data=(x_valid[valid_idx], y_valid[valid_idx]))
    model.save_history(history, name="train_" + str(seed) + "_" + str(size) + "_" + "classifier_his")

    history = model.train_classifier(y_train[train_idx], epochs=500, batch_size=batch_size, learning_rate=0.001,
                                     validation_data=(x_valid[valid_idx], y_valid[valid_idx]))
    model.save_history(history, name="train_" + str(seed) + "_" + str(size) + "_" + "classifier_his1")

    loss, accuracy = model.classifier.evaluate(x_valid[valid_idx], y_valid[valid_idx])
    if (accuracy > 0.75 and accuracy < 0.78) or (accuracy > 0.81 and accuracy < 0.83)  or (accuracy > 0.85 and accuracy < 0.89):
        compressed_subtrain = model.compressor.predict(extracted_train_features[train_idx], batch_size=batch_size, verbose=1)
        pd.DataFrame(np.append(compressed_subtrain, y_test, axis=1)).to_csv(
            os.path.join(os.getcwd(), "datasets", "subsets", "compressed_train_" + str(seed) + "_" + str(size) + "_" + str(accuracy) +".csv"), index=False)

    return accuracy

y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

# split a validation set
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
y_train = np.array(y_train)
y_test= np.array(y_test)
y_valid = np.array(y_valid)




print("There are {} training samples and {} validation samples".format(x_train.shape[0], x_valid.shape[0]))
print("There are {} test samples.".format(x_test.shape[0]))

# extract features
# features_train = model.extract(x_train, batch_size=batch_size)
# print("The shape of the extracted training sample features is: ", features_train.shape)

# save features
# model.save_features()

# load features
model.load_features()
extracted_train_features = model.extracted_features

test_scores = []
for j in range(1, 6):
    scores = []
    for i in range(1, 10):
        scores.append(subset(j*42, i*10))
    test_scores.append(scores)

np.save(os.path.join(os.getcwd(), "datasets", "subsets", "test_scores.npy"), test_scores)


# save trained model
# model.save_classifier()
# model.save_extractor()
#
# model.load_classifier()
# model.load_extractor()
# model.extract(model.features, y_train, batch_size=batch_size, compression=True)
#
# model.save_features()
#
# compressed_valid100 = model.compressor.predict(x_valid, batch_size=batch_size, verbose=1)
# compressed_test100 = model.compressor.predict(x_test, batch_size=batch_size, verbose=1)
#
# pd.DataFrame(np.append(compressed_valid100, y_valid, axis=1)).to_csv(os.path.join(os.getcwd(), "datasets", "cifar100", "compressed_valid.csv"),
#                                                                index=False)
# pd.DataFrame(np.append(compressed_test100, y_test, axis=1)).to_csv(os.path.join(os.getcwd(), "datasets", "cifar100", "compressed_test.csv"),
#                                                                index=False)

# Random Search for best fine tine hyper parameters
# rds = RandomSearch(model)
# best_dict, history_dict = rds(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size)
#
# model.save_history(history_dict, name="random_search_his")

# fine-tune the network
# print("Start to fine tune the network and extract compressed features.")
# history = model.fine_tune_features(x_train, y_train, learning_rate=best_dict["learning_rate"],
#                                    weight_decay=best_dict["weight_decay"], batch_size=batch_size, epochs=128,
#                                    validation_data=(x_valid, y_valid), early_stop=True)
# features = model.extract(x_train, compression=True)

# save results
# model.save_classifier()
# model.save_extractor()
#
# model.save_features()
#
# model.save_history(history, "fine_tune_his")
