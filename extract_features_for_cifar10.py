# check gpu devices
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd
import numpy as np
# load cifar10 datasets
from lib.data_loader import load_cifar10
from lib.feature_extractor import NASNetLargeExtractor

devices = tf.config.list_physical_devices('GPU')
if len(devices) < 1:
    raise AttributeError("No GPU found!")
else:
    print(devices)
    print()

batch_size = 128

# download google nasnet large pre-trained model
model = NASNetLargeExtractor(32, 10, model_path="models/cifar10", data_path="datasets/cifar10")
print("Pre-trained NASNetLarge is loaded.")

# preprocess the dataset
(x_train, y_train), (x_test, y_test) = load_cifar10()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


def preprocess_data(data_set):
    data_set /= 255.0
    return data_set


x_train = preprocess_data(x_train)
x_test = preprocess_data(x_test)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# split a validation set
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print("There are {} training samples and {} validation samples".format(x_train.shape[0], x_valid.shape[0]))
print("There are {} test samples.".format(x_test.shape[0]))

# extract features
features_train = model.extract(x_train, batch_size=batch_size)
print("The shape of the extracted training sample features is: ", features_train.shape)

# save features
model.save_features()

# load features
model.load_features()

# use dense layer to test feature quality
history = model.train_classifier(y_train, epochs=500, batch_size=batch_size, validation_data=(x_valid, y_valid))
model.save_history(history, name="train_classifier_his")

history = model.train_classifier(y_train, epochs=500, batch_size=batch_size, learning_rate=0.001, validation_data=(x_valid, y_valid))
model.save_history(history, name="train_classifier_his1")

# save trained model
model.save_classifier()
model.save_extractor()

model.load_classifier()
model.load_extractor()
model.load_features()
model.extract(model.features, y_train, batch_size=batch_size, compression=True)

model.save_features()

# compress validation, test data
compressed_valid10 = model.compressor.predict(x_valid, batch_size=batch_size, verbose=1)
compressed_test10 = model.compressor.predict(x_test, batch_size=batch_size, verbose=1)

pd.DataFrame(np.append(compressed_valid10, y_valid, axis=1)).to_csv(os.path.join(os.getcwd(), "datasets", "cifar10", "compressed_valid.csv"),
                                                               index=False)
pd.DataFrame(np.append(compressed_test10, y_test, axis=1)).to_csv(os.path.join(os.getcwd(), "datasets", "cifar10", "compressed_test.csv"),
                                                               index=False)





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
