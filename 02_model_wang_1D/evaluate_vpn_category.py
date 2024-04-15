import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchsummary import summary
import pickle
import csv
import pandas as pd
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    balanced_accuracy_score,
)
from util import load_model
from util import load_dataset
import os
from time import time
import warnings
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from util import plot

warnings.filterwarnings("always")


dataset_path = "dataset"
model_path = "model"
task = "category"
exp_name = "WANG1D_vpn_" + task
result_path = "result"
num_epochs = 1000
num_classes = 6
lr = 1e-3

print(">> load dataset: START")
X_train, Y_train, X_test, Y_test = load_dataset.load_dataset(task)
print(">> load dataset: END")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

print(">> load model: START")
model = load_model.CNN2D(num_classes=num_classes).to("cuda:0")
model.load_state_dict(torch.load(os.path.join(model_path, f"{exp_name}.pth")))
print(">> load model: END")

# summary(model, input_size=(1, 28, model_WANG_application28))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

y_true = []
y_prob = []


def plot_multiclass_reliability_diagram(y_true, y_prob, fname):
    n_classes = len(set(y_true))
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        prob_true, prob_pred = calibration_curve(
            [1 if cls == i else 0 for cls in y_true],
            [prob[i] for prob in y_prob],
            n_bins=100,
            strategy="uniform",
        )
        plt.plot(prob_pred, prob_true, marker="o", linestyle="--", label=f"Class {i}")
    plt.plot([0, 1], [0, 1], linestyle="-", color="gray")
    plt.xlim((0.6, 1))
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Reliability Diagram for Multiclass")
    plt.legend()
    plt.savefig(fname)


y_prob = None


# Test loop
def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        start = time()
        outputs = model(X_test)
        end = time() - start
        _, predicted = torch.max(outputs, 1)

        result_arr = []

        y_prob = outputs.cpu().numpy()

        for idx, i in enumerate(outputs.cpu()):
            prob = np.max(i.cpu().numpy()) / 10

            result_arr.append(
                [
                    predicted[idx].cpu().numpy(),
                    y_test[idx].cpu().numpy(),
                    round(prob, 4),
                ]
            )
        print(f"inf_time:{end}")
        pd.DataFrame(result_arr, columns=["predicted", "real", "prob"]).to_csv(
            os.path.join(result_path, f"{exp_name}.csv")
        )
    return y_prob


# Test loop
def test_model_plot(model, X_train, Y_train):
    model.eval()
    with torch.no_grad():
        start = time()
        outputs = model(X_train)
        end = time() - start
        _, predicted = torch.max(outputs, 1)

        result_arr = []

        for idx, i in enumerate(outputs.cpu()):
            prob = np.max(i.cpu().numpy()) / 10

            result_arr.append(
                [
                    predicted[idx].cpu().numpy(),
                    Y_train[idx].cpu().numpy(),
                    round(prob, 4),
                ]
            )

        y_true_train = Y_train.cpu().numpy()
        y_prob_train = outputs.cpu().numpy() / 10

        print(y_true_train.shape)
        print(y_prob_train.shape)
        plot.plot_multiclass_reliability_diagram(
            y_true_train, y_prob_train, f"1DCNN_{task}.png", task
        )


test_model(model, X_test, Y_test)
test_model_plot(model, X_train, Y_train)
