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
from util import config

warnings.filterwarnings("always")


def calculate_slope(point1, point2):
    """Calculate the slope between two points."""
    x1, y1 = point1
    x2, y2 = point2
    if x2 - x1 == 0:
        return float("inf")  # Return infinity for vertical lines
    return (y2 - y1) / (x2 - x1)


def calculate_slope(point1, point2):
    """Calculate the slope between two points."""
    x1, y1 = point1
    x2, y2 = point2
    if x2 - x1 == 0:
        return None  # Return None for vertical lines
    return (y2 - y1) / (x2 - x1)


def calculate_intercept(point, slope):
    """Calculate the y-intercept of a line passing through a point with given slope."""
    x, y = point
    if slope is None:
        return None  # Vertical line has no y-intercept
    return y - slope * x


def find_segment(x, points):
    """Find the segment that contains the given x value."""
    for i in range(len(points) - 1):
        x1, _ = points[i]
        x2, _ = points[i + 1]
        if x1 <= x <= x2 or x2 <= x <= x1:
            return i
    return None


def find_y(x, points):
    """Find the y value for a given x value."""
    segment_index = find_segment(x, points)
    if segment_index is None:
        return None  # No segment found for the given x value
    point1, point2 = points[segment_index], points[segment_index + 1]
    slope = calculate_slope(point1, point2)
    intercept = calculate_intercept(point1, slope)
    return slope * x + intercept


def find_steep_segments(points, threshold):
    """Find segments with steep slopes in a set of points."""
    steep_segments = []
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float("inf")
        if slope < threshold:
            steep_segments.append((points[i - 1], points[i]))
    return steep_segments


def plot_multiclass_reliability_diagram(y_true, y_prob, fname, task):

    print(y_true.shape)

    n_classes = len(set(y_true))
    plt.figure(figsize=(10, 8))

    integral_arr = [0] * 100

    for idx, i in enumerate(range(n_classes)):
        # if idx == 5:
        prob_true, prob_pred = calibration_curve(
            [1 if cls == i else 0 for cls in y_true],
            [prob[i] for prob in y_prob],
            n_bins=10,
            strategy="uniform",
        )

        temp_arr = []
        temp_arr.append((0, 0))
        for prob_idx, x in enumerate(prob_pred):
            temp_arr.append(
                (
                    prob_pred[prob_idx],
                    prob_true[prob_idx],
                )
            )
        temp_arr.append((1, 1))

        for ia_idx, ia in enumerate(integral_arr):
            y_value = find_y(ia_idx * 0.01, temp_arr)
            integral_arr[ia_idx] += y_value

        label = config.label[task]

        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            linestyle="--",
            label=f"{label[idx]}",
            alpha=0.5,
        )
    x_arr = []
    y_arr = []
    slope_arr = []

    for idx, i in enumerate(integral_arr):
        x_arr.append(idx * 0.01)
        y_arr.append(i)

    for i in range(len(integral_arr) - 1):
        slope_arr.append(
            calculate_slope((x_arr[i], y_arr[i]), (x_arr[i + 1], y_arr[i + 1]))
            / (n_classes * 1.5)
        )

    # plt.plot(
    #     x_arr[:-1],
    #     slope_arr,
    #     # marker="*",
    #     linestyle="-",
    #     label=f"slope",
    #     alpha=1.0,
    # )
    temp_asdfad = np.array([x_arr[:-1], slope_arr])

    steep_segments = find_steep_segments(temp_asdfad.T, -5)

    # Print the found steep segments
    for segment in steep_segments:
        x_point = segment[0][0]
        if x_point > 0.5:
            plt.plot(
                [x_point, x_point],
                [1.1, -0.1],
                color="r",
                linestyle="--",
                # label=f"slope",
                alpha=0.3,
            )
            print(f"tau : {x_point}")

    # plt.set_ylabel("Label", labelpad=20, rotation=0)
    plt.plot([0, 1], [0, 1], linestyle="-", color="gray")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Reliability Diagram(1DCNN)")
    plt.legend(loc="upper left")
    plt.savefig(fname)
