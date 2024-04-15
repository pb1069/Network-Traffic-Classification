from keras import backend as K
from keras.models import Model
from keras.layers import (
    Dense,
    Input,
    Dropout,
    MaxPooling1D,
    Conv1D,
    GlobalMaxPool1D,
    Activation,
)
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.utils import np_utils
import keras.backend as K
import numpy as np
import tensorflow as tf
import keras.callbacks
import sys
import os
from timeit import default_timer as timer
import glob
import time
import pickle
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import math
import pandas as pd
import os
from util import load_model, load_dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

dataset_path = "dataset"
model_path = "model"
task = "encapsulation"
exp_name = "HASTIDS2_vpn_" + task
result_path = "result"
num_epochs = 50
num_classes = 2
lr = 1e-3


class Dataloader(Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    # batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
        # sampler의 역할(index를 batch_size만큼 sampling해줌)
        indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        return np.array(batch_x), np.array(batch_y)

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


def one_hot_encode(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    one_hot_labels = np.zeros((num_labels, num_classes))
    one_hot_labels.flat[index_offset + labels.ravel()] = 1
    return one_hot_labels


print(">> load dataset: START")
X_train, Y_train, X_test, Y_test = load_dataset.load_dataset(task)
X_train = np.reshape(X_train, (-1, 6, 100))
X_test = np.reshape(X_test, (-1, 6, 100))
Y_train = one_hot_encode(Y_train, num_classes)
Y_test = one_hot_encode(Y_test, num_classes)
train_loader = Dataloader(X_train, Y_train, 128, shuffle=True)
test_loader = Dataloader(X_test, Y_test, 128)
print(">> load dataset: END")

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

print(">> load model: START")
model = load_model.load_HASTIDS2(100, num_classes)
optimizer = keras.optimizers.Adam(learning_rate=lr)
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["acc"],
)
print(">> load model: END")

model = keras.models.load_model(os.path.join(model_path, f"{exp_name}.pth"))
from time import time

start = time()
pred = model.predict(test_loader)
end = time() - start
print(f"inf_time:{end}")

result_arr = []
for idx, i in enumerate(pred):
    real_class = np.argmax(Y_test[idx])
    pred_class = np.argmax(pred[idx])
    prob = pred[idx][np.argmax(pred[idx])]

    result_arr.append([pred_class, real_class, prob])

pd.DataFrame(result_arr, columns=["predicted", "real", "prob"]).to_csv(
    os.path.join(result_path, f"{exp_name}.csv")
)
