from lib.experiments import load_dataset, run_wcl, train_with_original, run_pop, run_egdis, run_cl, run_wcl2, run_wcl3
from lib.feature_extractor import NASNetLargeExtractor

import os
import pandas as pd

batch_size = 256

model10 = NASNetLargeExtractor(32, 10, model_path="models/cifar10", data_path="datasets/cifar10")
model100 = NASNetLargeExtractor(32, 100, model_path="models/cifar100", data_path="datasets/cifar100")

print("Pre-trained NASNetLarge is loaded.")

model10.load_classifier()
model100.load_classifier()

# load compressed dataset
x_train10, x_valid10, x_test10, y_train10, y_valid10, y_test10 = load_dataset("cifar10")
print("cifar10 loaded")
x_train100, x_valid100, x_test100, y_train100, y_valid100, y_test100 = load_dataset("cifar100")
print("cifar100 loaded")

def preprocess_data(data_set):
    data_set /= 255.0
    return data_set

x_train10 = x_train10.astype('float32')
x_valid10 = x_valid10.astype('float32')
x_test10 = x_test10.astype('float32')

x_train100 = x_train100.astype('float32')
x_valid100 = x_valid100.astype('float32')
x_test100 = x_test100.astype('float32')

x_train10 = preprocess_data(x_train10)
x_valid10 = preprocess_data(x_valid10)
x_test10 = preprocess_data(x_test10)

x_train100 = preprocess_data(x_train100)
x_valid100 = preprocess_data(x_valid100)
x_test100 = preprocess_data(x_test100)

print("pre-process done")

# compress validation, test data
compressed_valid10 = model10.compressor.predict(x_valid10, batch_size=batch_size, verbose=1)
compressed_test10 = model10.compressor.predict(x_test10, batch_size=batch_size, verbose=1)

compressed_valid100 = model100.compressor.predict(x_valid100, batch_size=batch_size, verbose=1)
compressed_test100 = model100.compressor.predict(x_test100, batch_size=batch_size, verbose=1)

pd.DataFrame(compressed_valid10).to_csv(os.path.join(os.getcwd(), "datasets", "cifar10", "compressed_valid.csv"),
                                                               index=False)
pd.DataFrame(compressed_test10).to_csv(os.path.join(os.getcwd(), "datasets", "cifar10", "compressed_test.csv"),
                                                               index=False)


pd.DataFrame(compressed_valid100).to_csv(os.path.join(os.getcwd(), "datasets", "cifar100", "compressed_valid.csv"),
                                                               index=False)
pd.DataFrame(compressed_test100).to_csv(os.path.join(os.getcwd(), "datasets", "cifar100", "compressed_test.csv"),
                                                               index=False)