import os
import numpy as np
import torch

dataset_path = "dataset"
model_path = "model"
exp_name = "SAM_vpn_application"
result_path = "result"


def load_dataset(task):
    # load train dataset

    if task == "application":
        X_SAM_train = np.load(
            os.path.join(dataset_path, f"X_SAM_application_train.npy")
        )
        Y_SAM_train = np.load(
            os.path.join(dataset_path, f"Y_SAM_application_train.npy")
        )
        X_SAM_test = np.load(os.path.join(dataset_path, f"X_SAM_application_test.npy"))
        Y_SAM_test = np.load(os.path.join(dataset_path, f"Y_SAM_application_test.npy"))

    elif task == "category":
        X_SAM_train = np.load(os.path.join(dataset_path, f"X_SAM_category_train.npy"))
        Y_SAM_train = np.load(os.path.join(dataset_path, f"Y_SAM_category_train.npy"))

        X_SAM_test = np.load(os.path.join(dataset_path, f"X_SAM_category_test.npy"))
        Y_SAM_test = np.load(os.path.join(dataset_path, f"Y_SAM_category_test.npy"))
    else:
        X_SAM_train = np.load(
            os.path.join(dataset_path, f"X_SAM_encapsulation_train.npy")
        )
        Y_SAM_train = np.load(
            os.path.join(dataset_path, f"Y_SAM_encapsulation_train.npy")
        )

        X_SAM_test = np.load(
            os.path.join(dataset_path, f"X_SAM_encapsulation_test.npy")
        )
        Y_SAM_test = np.load(
            os.path.join(dataset_path, f"Y_SAM_encapsulation_test.npy")
        )

    X_SAM_train = X_SAM_train * 256
    X_SAM_test = X_SAM_test * 256

    X_pos_train = []
    for _ in range(X_SAM_train.shape[0]):
        X_pos_train.append(np.arange(50))
    X_pos_train = np.array(X_pos_train)

    X_pos_test = []
    for _ in range(X_SAM_test.shape[0]):
        X_pos_test.append(np.arange(50))
    X_pos_test = np.array(X_pos_test)

    X_train = torch.tensor(X_SAM_train, dtype=torch.int64).to("cuda:0")
    Y_train = torch.tensor(Y_SAM_train, dtype=torch.long).to("cuda:0")
    X_test = torch.tensor(X_SAM_test, dtype=torch.int64).to("cuda:0")
    Y_test = torch.tensor(Y_SAM_test, dtype=torch.long).to("cuda:0")

    X_pos_train = torch.tensor(X_pos_train, dtype=torch.int64).to("cuda:0")
    X_pos_test = torch.tensor(X_pos_test, dtype=torch.int64).to("cuda:0")

    return X_train, Y_train, X_test, Y_test, X_pos_train, X_pos_test
