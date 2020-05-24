import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp


log_dir = "logs/hparam_tuning_small_2lyrs/"

combined = pd.read_csv(os.path.join("formatted_data", "combined.csv"))

# Seed the random generator, so we get the same results each time
random_seed_1 = 3141592
random_seed_2 = 8675309

# Specify which column contains the class label (Success or Not)
class_label_col = "offer_successful"

# Split the data and class labels, so we can use train_test_split.
data_no_label = combined.drop(class_label_col, axis=1)
class_labels = combined.loc[:, [class_label_col]]

# We want 80/20, and then we will split the training set into validation.
(X_train, X_test, y_train, y_test) = train_test_split(
    data_no_label, class_labels, test_size=0.2, random_state=random_seed_1
)

# Split into training/validation.
# To get 60/20/20 split from 80, we use a test size of 0.25 since 60/80 = 0.75.
(X_train, X_valid, y_train, y_valid) = train_test_split(
    X_train, y_train, test_size=0.25, random_state=random_seed_2
)

# Need to change the shape of the y values to be (n, ).
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
y_valid = y_valid.values.ravel()

# Since we want to be able to create new offers to add to our portfolio, we don't want to associate
# too much importance to the exact id_offer. We want to generalize.

# Therefore, we will store the id_offer's for each training set in case we need it later.
id_offer_train = X_train.loc[:, "id_offer"]
id_offer_test = X_test.loc[:, "id_offer"]
id_offer_valid = X_valid.loc[:, "id_offer"]

# Now remove 'id_offer' from each.
X_train = X_train.drop(["id_offer"], axis=1)
X_test = X_test.drop(["id_offer"], axis=1)
X_valid = X_valid.drop(["id_offer"], axis=1)

# Remove any logs from previous runs
os.system("rm -rf ./" + log_dir)


def f_beta_score(matrix, beta):
    """Calculate the f_beta score.

    Args:
        -matrix (1d array) = [tn fp fn tp]
        -beta (int) = indicates if we use F1 score, F2 score, etc.

    Returns:
        -score (float)
    """
    tn, fp, fn, tp = matrix

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    score = (
        (1 + beta ** 2) * (precision * recall) / (((beta ** 2) * precision) + recall)
    )

    return score


def f1_score(matrix):
    """Calculate F1 score"""
    return f_beta_score(matrix, 1)


def f2_score(matrix):
    """Calculate F2 score"""
    return f_beta_score(matrix, 2)


# List of hyperparameters
HP_NUM_UNITS_1 = hp.HParam("num_units_1", hp.Discrete([32, 64, 128]))
HP_DROPOUT_1 = hp.HParam("dropout_1", hp.RealInterval(0.1, 0.5))
HP_NUM_LAYERS = hp.HParam(
    "num_layers", hp.Discrete([2])
)  # hp.HParam("num_layers", hp.Discrete([1, 2]))
HP_NUM_UNITS_2 = hp.HParam(
    "num_units_2", hp.Discrete([32, 64, 128])
)  # hp.HParam("num_units_2", hp.Discrete([32, 64, 128]))
HP_DROPOUT_2 = hp.HParam(
    "dropout_2", hp.RealInterval(0.1, 0.5)
)  # hp.HParam("dropout_2", hp.RealInterval(0.1, 0.5))
HP_OPTIMIZER = hp.HParam(
    "optimizer", hp.Discrete(["adam"])
)  # hp.HParam("optimizer", hp.Discrete(["adam", "sgd"]))
HP_OPTIM_LR = hp.HParam("lr", hp.Discrete([1e-4, 1e-3]))

METRIC_ACCURACY = "accuracy"
METRIC_FN = "false_negatives"  # tf.keras.metrics.FalseNegatives()
METRIC_FP = "false_positives"  # tf.keras.metrics.FalsePositives()
METRIC_TN = "true_negatives"  # tf.keras.metrics.TrueNegatives()
METRIC_TP = "true_positives"  # tf.keras.metrics.TruePositives()
METRIC_F1 = "f1_score"
METRIC_F2 = "f2_score"

with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
        hparams=[
            HP_NUM_UNITS_1,
            HP_DROPOUT_1,
            HP_NUM_UNITS_2,
            HP_DROPOUT_2,
            HP_OPTIMIZER,
            HP_OPTIM_LR,
        ],
        metrics=[
            hp.Metric(METRIC_ACCURACY, display_name="Accuracy"),
            hp.Metric(METRIC_FN, display_name="False Negatives"),
            hp.Metric(METRIC_FP, display_name="False Positives"),
            hp.Metric(METRIC_TN, display_name="True Negatives"),
            hp.Metric(METRIC_TP, display_name="True Positives"),
            hp.Metric(METRIC_F1, display_name="F1 Score"),
            hp.Metric(METRIC_F2, display_name="F2 Score"),
        ],
    )


def validate_model(hparams):
    """Perform hyperparameter tuning on the validation set."""
    model = keras.Sequential()
    # First hidden layer.
    model.add(
        keras.layers.Dense(
            hparams[HP_NUM_UNITS_1], activation="relu", input_shape=[X_train.shape[1]]
        )
    )
    model.add(keras.layers.Dropout(hparams[HP_DROPOUT_1]))
    # Possibly add second hidden layer.
    if hparams[HP_NUM_LAYERS] == 2:
        model.add(keras.layers.Dense(hparams[HP_NUM_UNITS_2], activation="relu"))
        model.add(keras.layers.Dropout(hparams[HP_DROPOUT_2]))
    # Output Layer
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Display model summary
    model.summary()

    # Initialize optimizer with learning rate.
    if hparams[HP_OPTIMIZER] == "adam":
        optim = keras.optimizers.Adam(learning_rate=hparams[HP_OPTIM_LR])
    elif hparams[HP_OPTIMIZER] == "sgd":
        optim = keras.optimizers.SGD(learning_rate=hparams[HP_OPTIM_LR])

    # Compile the model.
    model.compile(
        optimizer=optim,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.TruePositives(),
        ],
    )

    # Callbacks
    # Early Stopping
    #   -monitor validation loss.
    #   -when validation loss stops decreasing, stop.
    #   -patience is number of epochs with no improvement.
    cb_es = keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=20, verbose=2
    )
    # Model Checkpoint
    #   -call our model "best_model.h5".
    #   -monitor validation loss.
    #   -when validation loss stops decreasing, stop.
    #   -save the best overall model.
    cb_ckpt = keras.callbacks.ModelCheckpoint(
        "best_model_small_2lyr.h5",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=2,
    )

    # Fit
    model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        callbacks=[cb_es, cb_ckpt],
        epochs=200,
        verbose=2,
    )

    _, test_acc, test_fn, test_fp, test_tn, test_tp = model.evaluate(
        X_test, y_test, verbose=2
    )

    return test_acc, test_fn, test_fp, test_tn, test_tp


# For each run, log an hparams summary with hyperparameters and metrics.
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial.
        accuracy, fn, fp, tn, tp = validate_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        tf.summary.scalar(METRIC_FN, fn, step=1)
        tf.summary.scalar(METRIC_FP, fp, step=1)
        tf.summary.scalar(METRIC_TN, tn, step=1)
        tf.summary.scalar(METRIC_TP, tp, step=1)
        tf.summary.scalar(METRIC_F1, f1_score(np.array([tn, fp, fn, tp])), step=1)
        tf.summary.scalar(METRIC_F2, f2_score(np.array([tn, fp, fn, tp])), step=1)


# Grid search over parameters and log values.
session_num = 0

for num_units_1 in HP_NUM_UNITS_1.domain.values:
    for dropout_1 in np.arange(
        HP_DROPOUT_1.domain.min_value, HP_DROPOUT_1.domain.max_value + 0.1, 0.2
    ):
        for num_layers in HP_NUM_LAYERS.domain.values:
            for num_units_2 in HP_NUM_UNITS_2.domain.values:
                for dropout_2 in np.arange(
                    HP_DROPOUT_2.domain.min_value,
                    HP_DROPOUT_2.domain.max_value + 0.1,
                    0.2,
                ):
                    for optimizer in HP_OPTIMIZER.domain.values:
                        for optim_lr in HP_OPTIM_LR.domain.values:
                            hparams = {
                                HP_NUM_UNITS_1: num_units_1,
                                HP_DROPOUT_1: dropout_1,
                                HP_NUM_LAYERS: num_layers,
                                HP_NUM_UNITS_2: num_units_2,
                                HP_DROPOUT_2: dropout_2,
                                HP_OPTIMIZER: optimizer,
                                HP_OPTIM_LR: optim_lr,
                            }
                            run_name = "run-%d" % session_num
                            print(f"--- Starting trial: {run_name}")
                            print({h.name: hparams[h] for h in hparams})
                            run(log_dir + run_name, hparams)
                            session_num += 1
