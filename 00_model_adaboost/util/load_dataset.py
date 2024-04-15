import os
import numpy as np

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
        # load test dataset
        X_WANG_test = np.load(
            os.path.join(dataset_path, f"X_WANG_application_test.npy")
        )
        Y_WANG_test = np.load(
            os.path.join(dataset_path, f"Y_WANG_application_test.npy")
        )

    elif task == "category":
        X_WANG_train = np.load(os.path.join(dataset_path, f"X_WANG_category_train.npy"))
        Y_WANG_train = np.load(os.path.join(dataset_path, f"Y_WANG_category_train.npy"))

        # load test dataset
        X_WANG_test = np.load(os.path.join(dataset_path, f"X_WANG_category_test.npy"))
        Y_WANG_test = np.load(os.path.join(dataset_path, f"Y_WANG_category_test.npy"))
    else:
        X_WANG_train = np.load(
            os.path.join(dataset_path, f"X_WANG_encapsulation_train.npy")
        )
        Y_WANG_train = np.load(
            os.path.join(dataset_path, f"Y_WANG_encapsulation_train.npy")
        )

        # load test dataset
        X_WANG_test = np.load(
            os.path.join(dataset_path, f"X_WANG_encapsulation_test.npy")
        )
        Y_WANG_test = np.load(
            os.path.join(dataset_path, f"Y_WANG_encapsulation_test.npy")
        )

    X_train = X_WANG_train
    Y_train = Y_WANG_train
    X_test = X_WANG_test
    Y_test = Y_WANG_test

    return X_train, Y_train, X_test, Y_test
