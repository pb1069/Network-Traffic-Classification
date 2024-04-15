import os
import numpy as np

dataset_path = "dataset"
model_path = "model"
exp_name = "HASTIDS1_vpn_application"
result_path = "result"


def load_dataset(task):
    # load train dataset

    if task == "application":
        X_HASTIDS1_train = np.load(
            os.path.join(dataset_path, f"X_HASTIDS1_application_train.npy")
        )
        Y_HASTIDS1_train = np.load(
            os.path.join(dataset_path, f"Y_HASTIDS1_application_train.npy")
        )

        # load test dataset
        X_HASTIDS1_test = np.load(
            os.path.join(dataset_path, f"X_HASTIDS1_application_test.npy")
        )
        Y_HASTIDS1_test = np.load(
            os.path.join(dataset_path, f"Y_HASTIDS1_application_test.npy")
        )

    elif task == "category":
        X_HASTIDS1_train = np.load(
            os.path.join(dataset_path, f"X_HASTIDS1_category_train.npy")
        )
        Y_HASTIDS1_train = np.load(
            os.path.join(dataset_path, f"Y_HASTIDS1_category_train.npy")
        )

        # load test dataset
        X_HASTIDS1_test = np.load(
            os.path.join(dataset_path, f"X_HASTIDS1_category_test.npy")
        )
        Y_HASTIDS1_test = np.load(
            os.path.join(dataset_path, f"Y_HASTIDS1_category_test.npy")
        )
    else:
        X_HASTIDS1_train = np.load(
            os.path.join(dataset_path, f"X_HASTIDS1_encapsulation_train.npy")
        )
        Y_HASTIDS1_train = np.load(
            os.path.join(dataset_path, f"Y_HASTIDS1_encapsulation_train.npy")
        )

        # load test dataset
        X_HASTIDS1_test = np.load(
            os.path.join(dataset_path, f"X_HASTIDS1_encapsulation_test.npy")
        )
        Y_HASTIDS1_test = np.load(
            os.path.join(dataset_path, f"Y_HASTIDS1_encapsulation_test.npy")
        )

    X_HASTIDS1_train = X_HASTIDS1_train * 256
    X_HASTIDS1_test = X_HASTIDS1_test * 256

    Y_HASTIDS1_train = Y_HASTIDS1_train
    Y_HASTIDS1_test = Y_HASTIDS1_test

    return X_HASTIDS1_train, Y_HASTIDS1_train, X_HASTIDS1_test, Y_HASTIDS1_test
