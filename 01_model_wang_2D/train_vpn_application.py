import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchsummary import summary
import pickle
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    balanced_accuracy_score,
)
import os
import warnings
from util import load_model
from util import load_dataset
from time import time

warnings.filterwarnings("always")


dataset_path = "dataset"
model_path = "model"
task = "application"
exp_name = "WANG2D_vpn_" + task
result_path = "result"
num_epochs = 1000
num_classes = 20
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
print(">> load model: END")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Training loop
def train_model(model, optimizer, criterion, X_train, y_train, num_epochs=num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Convert outputs to labels
        _, predicted = torch.max(outputs, 1)

        # Calculate metrics
        accuracy = balanced_accuracy_score(y_train.cpu(), predicted.cpu())
        f1 = f1_score(
            y_train.cpu(), predicted.cpu(), average="macro", zero_division=np.nan
        )
        recall = recall_score(
            y_train.cpu(), predicted.cpu(), average="macro", zero_division=np.nan
        )
        precision = precision_score(
            y_train.cpu(), predicted.cpu(), average="macro", zero_division=np.nan
        )

        # Print metrics
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {round(loss.item(),7)}, Accuracy: {round(accuracy,4)}, F1-Score: {round(f1,4)}, Recall: {round(recall,4)} , Precision: {round(precision,4)}"
        )


# Test loop
def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        start = time()
        outputs = model(X_test)
        end = time() - start
        _, predicted = torch.max(outputs, 1)

        result_arr = []

        for idx, i in enumerate(outputs.cpu()):
            prob = np.max(i.cpu().numpy())

            result_arr.append(
                [predicted[idx].cpu().numpy(), y_test[idx].cpu().numpy(), prob]
            )

        # Convert outputs to labels
        _, predicted = torch.max(outputs, 1)

        # Calculate metrics
        accuracy = balanced_accuracy_score(y_test.cpu(), predicted.cpu())
        f1 = f1_score(
            y_test.cpu(), predicted.cpu(), average="macro", zero_division=np.nan
        )
        recall = recall_score(
            y_test.cpu(), predicted.cpu(), average="macro", zero_division=np.nan
        )
        precision = precision_score(
            y_test.cpu(), predicted.cpu(), average="macro", zero_division=np.nan
        )

        # Print metrics
        print(
            f"TEST: Accuracy: {round(accuracy,4)}, F1-Score: {round(f1,4)}, Recall: {round(recall,4)} , Precision: {round(precision,4)}"
        )


print(">> Train : START")
train_model(model, optimizer, criterion, X_train, Y_train)
print(">> Train : END")
test_model(model, X_test, Y_test)
print(">> Save model : START")
torch.save(model.state_dict(), os.path.join(model_path, f"{exp_name}.pth"))
print(">> Save model : END")
