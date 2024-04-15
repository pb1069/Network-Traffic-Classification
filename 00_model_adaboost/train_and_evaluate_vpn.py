import numpy as np
import pickle
import csv
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    balanced_accuracy_score,
)
from util import load_dataset
import os
from time import time
import warnings
from sklearn.ensemble import AdaBoostClassifier
import sys
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from util import plot

warnings.filterwarnings("always")

dataset_path = "dataset"
model_path = "model"
task = sys.argv[1]
exp_name = "AB_vpn_" + task
result_path = "result"
num_epochs = 1000
num_classes = int(sys.argv[2])
lr = 1e-3

print(">> load dataset: START")
X_train, Y_train, X_test, Y_test = load_dataset.load_dataset(task)
print(">> load dataset: END")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

new_X_train = X_train[:, :50]
new_X_test = X_test[:, :50]

model = xgb.XGBClassifier()
print(model)
# 모델 훈련
model.fit(new_X_train, Y_train)


start = time()
Y_pred = model.predict_proba(new_X_test)
end = time() - start

y_true = Y_train
y_prob = []

result_arr = []
for idx, i in enumerate(Y_pred):
    y_prob.append(i)
    result_arr.append([np.argmax(i), Y_test[idx], np.max(i)])


## print test_result
pd.DataFrame(result_arr, columns=["predicted", "real", "prob"]).to_csv(
    os.path.join(result_path, f"{exp_name}.csv")
)

## print train_result
y_true_train = Y_train
y_prob_train = []

Y_pred_train = model.predict_proba(new_X_train)
result_arr = []
for idx, i in enumerate(Y_pred_train):
    y_prob_train.append(i)

plot.plot_multiclass_reliability_diagram(
    y_true_train, y_prob_train, f"RD_{task}.png", task
)

print(f"inf_time:{end}")
