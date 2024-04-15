import os
import numpy as np
import torch

dataset_path = "dataset"
model_path = "model"
exp_name = "WANG_vpn_application"
result_path = "result"


def load_dataset(task):
    # load train dataset

    if task == "application":
        X_WANG_train = np.load(
            os.path.join(dataset_path, f"X_WANG_application_train.npy")
        )
        Y_WANG_train = np.load(
            os.path.join(dataset_path, f"Y_WANG_application_train.npy")
        )
        X_train = torch.tensor(X_WANG_train, dtype=torch.float32).to("cuda:0")
        Y_train = torch.tensor(Y_WANG_train, dtype=torch.long).to("cuda:0")

        # load test dataset
        X_WANG_test = np.load(
            os.path.join(dataset_path, f"X_WANG_application_test.npy")
        )
        Y_WANG_test = np.load(
            os.path.join(dataset_path, f"Y_WANG_application_test.npy")
        )
        X_test = torch.tensor(X_WANG_test, dtype=torch.float32).to("cuda:0")
        Y_test = torch.tensor(Y_WANG_test, dtype=torch.long).to("cuda:0")

    elif task == "category":
        X_WANG_train = np.load(os.path.join(dataset_path, f"X_WANG_category_train.npy"))
        Y_WANG_train = np.load(os.path.join(dataset_path, f"Y_WANG_category_train.npy"))
        X_train = torch.tensor(X_WANG_train, dtype=torch.float32).to("cuda:0")
        Y_train = torch.tensor(Y_WANG_train, dtype=torch.long).to("cuda:0")

        # load test dataset
        X_WANG_test = np.load(os.path.join(dataset_path, f"X_WANG_category_test.npy"))
        Y_WANG_test = np.load(os.path.join(dataset_path, f"Y_WANG_category_test.npy"))
        X_test = torch.tensor(X_WANG_test, dtype=torch.float32).to("cuda:0")
        Y_test = torch.tensor(Y_WANG_test, dtype=torch.long).to("cuda:0")
    else:
        X_WANG_train = np.load(
            os.path.join(dataset_path, f"X_WANG_encapsulation_train.npy")
        )
        Y_WANG_train = np.load(
            os.path.join(dataset_path, f"Y_WANG_encapsulation_train.npy")
        )
        X_train = torch.tensor(X_WANG_train, dtype=torch.float32).to("cuda:0")
        Y_train = torch.tensor(Y_WANG_train, dtype=torch.long).to("cuda:0")

        # load test dataset
        X_WANG_test = np.load(
            os.path.join(dataset_path, f"X_WANG_encapsulation_test.npy")
        )
        Y_WANG_test = np.load(
            os.path.join(dataset_path, f"Y_WANG_encapsulation_test.npy")
        )
        X_test = torch.tensor(X_WANG_test, dtype=torch.float32).to("cuda:0")
        Y_test = torch.tensor(Y_WANG_test, dtype=torch.long).to("cuda:0")

    return X_train, Y_train, X_test, Y_test
