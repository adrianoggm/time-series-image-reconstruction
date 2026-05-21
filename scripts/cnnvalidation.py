#!/usr/bin/env python3
"""CNN baseline (paper-style 1D-Conv + statistical features) on WISDM windows."""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.layers import (Concatenate, Conv1D, Dense, Dropout,
                                     Flatten, Input, Lambda, MaxPooling1D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import src.ts_plots as plt
from src.config import DEFAULT_DATA_DIR
from src.paths import ALL_STYPES, xall_path, windowed_npy
from src.results import save_run_metrics

CLASSIFIER_NAME = "cnn"


def transformary(y):
    _y = np.ones(len(y)) * -1
    for i in range(len(y)):
        for j in range(y.shape[1]):
            if y[i, j] <= 1.1 and y[i, j] >= 0.9:
                _y[i] = j
    return _y


def compute_additional_features(x, bins=10):
    feats = []
    for ch in range(x.shape[1]):
        channel_data = x[:, ch]
        m = np.mean(channel_data)
        var = np.var(channel_data)
        s = np.sum(np.abs(channel_data))
        min_val, max_val = np.min(channel_data), np.max(channel_data)
        if min_val == max_val:
            hist = np.zeros(bins, dtype=np.float32)
        else:
            hist, _ = np.histogram(channel_data, bins=bins, range=(min_val, max_val))
            hist = hist.astype(np.float32)
        feats.extend([m, var, s])
        feats.extend(hist.tolist())
    return np.array(feats, dtype=np.float32)


def additional_features_layer(x, bins=10):
    def _compute_features(sample):
        return compute_additional_features(sample, bins=bins)
    features = tf.map_fn(lambda samp: tf.py_function(_compute_features, [samp], tf.float32),
                         x, fn_output_signature=tf.float32)
    features.set_shape([None, 3 * (3 + bins)])
    return features


def create_cnn_model_paper(input_shape, num_classes=6, l2_reg=0.001, bins=10):
    inp = Input(shape=input_shape)
    conv = Conv1D(filters=196, kernel_size=16, strides=1, activation="relu",
                  kernel_regularizer=l2(l2_reg))(inp)
    pool = MaxPooling1D(pool_size=4)(conv)
    flat = Flatten()(pool)
    stats = Lambda(lambda x: additional_features_layer(x, bins=bins))(inp)
    concat = Concatenate()([flat, stats])
    fc = Dense(1024, activation="relu", kernel_regularizer=l2(l2_reg))(concat)
    fc = Dropout(0.05)(fc)
    out = Dense(num_classes, activation="softmax")(fc)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _load_reconstruction(stype, sampling, fold, n_folds, dataset_folder):
    arr = np.load(xall_path(stype, sampling, fold, n_folds, "test", dataset_folder))
    return arr if stype == "RP" else np.transpose(arr, (0, 2, 1))


def run(
    stype: str,
    sampling: str = "loto",
    fold: int = 0,
    n_folds: int = 3,
    dataset_folder: str = DEFAULT_DATA_DIR,
    save_results: bool = True,
    model_path: str | None = None,
    train_new: bool = False,
    epochs: int = 20,
    batch_size: int = 64,
):
    if model_path is None:
        model_path = os.path.join(dataset_folder, "modelo_cnnPaperX.h5")

    X_train = np.load(windowed_npy("training_data",   sampling, fold, n_folds, dataset_folder))
    y_train = np.load(windowed_npy("y_training_data", sampling, fold, n_folds, dataset_folder))
    X_val   = np.load(windowed_npy("validation_data", sampling, fold, n_folds, dataset_folder))
    y_val   = np.load(windowed_npy("y_validation_data", sampling, fold, n_folds, dataset_folder))

    y_train_trans = transformary(y_train)
    y_val_trans   = transformary(y_val)

    if train_new or not os.path.exists(model_path):
        model = create_cnn_model_paper(input_shape=(X_train.shape[1], X_train.shape[2]),
                                       num_classes=6, l2_reg=0.0001, bins=10)
        print(model.summary())
        model.fit(X_train, y_train_trans, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_val, y_val_trans))
        model.save(model_path)
    else:
        model = load_model(model_path,
                           custom_objects={"compute_additional_features": compute_additional_features})

    pred_val = model.predict(X_val).argmax(axis=-1)
    original_report = classification_report(y_val_trans, pred_val, output_dict=True, zero_division=0)
    print(f"\n=== {CLASSIFIER_NAME} on original validation_data ({stype}) ===")
    plt.plot_cm(y_val_trans, pred_val)
    print(classification_report(y_val_trans, pred_val, zero_division=0))

    X_recon = _load_reconstruction(stype, sampling, fold, n_folds, dataset_folder)
    pred_recon = model.predict(X_recon).argmax(axis=-1)
    reconstructed_report = classification_report(y_val_trans, pred_recon, output_dict=True, zero_division=0)
    print(f"\n=== {CLASSIFIER_NAME} on {stype} reconstructions ===")
    plt.plot_cm(y_val_trans, pred_recon)
    print(classification_report(y_val_trans, pred_recon, zero_division=0))

    if save_results:
        out = save_run_metrics(
            stype=stype, classifier=CLASSIFIER_NAME,
            sampling=sampling, fold=fold, n_folds=n_folds,
            original_report=original_report,
            reconstructed_report=reconstructed_report,
            extra={"model_path": str(model_path), "train_new": train_new},
        )
        print(f"\nSaved metrics to {out}")
    return original_report, reconstructed_report


def build_argparser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stype", default="GAF", choices=ALL_STYPES)
    p.add_argument("--sampling", default="loto", choices=["loto", "loso"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--data-folder", default=DEFAULT_DATA_DIR)
    p.add_argument("--no-save-results", action="store_true")
    p.add_argument("--model-path", default=None)
    p.add_argument("--train-new", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    return p


def main():
    args = build_argparser().parse_args()
    run(
        stype=args.stype, sampling=args.sampling, fold=args.fold, n_folds=args.n_folds,
        dataset_folder=args.data_folder, save_results=not args.no_save_results,
        model_path=args.model_path, train_new=args.train_new,
        epochs=args.epochs, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
