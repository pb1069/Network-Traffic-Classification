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

LSTM_UNIT = 92


def binarize(x, sz=256):
    return tf.cast(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1), tf.float32)


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 256


def byte_block(
    in_layer,
    nb_filter=(64, 100),
    filter_length=(3, 3),
    subsample=(2, 1),
    pool_length=(2, 2),
):
    block = in_layer
    for i in range(len(nb_filter)):
        block = Conv1D(
            filters=nb_filter[i],
            kernel_size=filter_length[i],
            padding="valid",
            activation="tanh",
            strides=subsample[i],
        )(block)
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)

    block = GlobalMaxPool1D()(block)
    block = Dense(1024, activation="relu")(block)
    return block


def load_HASTIDS1(PACKET_LEN, num_classes):
    session = Input(shape=(PACKET_LEN), dtype="int64")
    embedded = Lambda(binarize, output_shape=binarize_outshape)(session)
    block2 = byte_block(
        embedded, (32, 64), filter_length=(5, 5), subsample=(1, 1), pool_length=(3, 3)
    )
    dense_layer = Dense(num_classes, name="dense_layer")(block2)
    output = tf.keras.activations.sigmoid(dense_layer)
    model = Model(outputs=output, inputs=session)

    return model
