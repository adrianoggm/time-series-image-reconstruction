#!/usr/bin/env python3
"""CNN baseline (paper-style 1D-Conv + statistical features) on WISDM windows.

The statistical-feature branch used to be ``tf.map_fn(py_function(...))`` which
runs sample-by-sample in Python and dominates train/inference time. We now
replace it with vectorised, GPU-friendly TF ops: mean, variance and sum-abs
per channel + a 10-bin histogram per channel computed by ``tf.histogram_fixed_width_bins``
on a fixed accelerometer range.

That gives identical-rank features (channels × (3 + bins) = 3 × 13 = 39 per
window) but removes the Python round-trip. Note: this changes the model
graph, so the cached ``models/cnn.h5`` produced by the legacy code is **not**
load-compatible — the evaluator will retrain on first use if no compatible
cache exists.
"""

from __future__ import annotations

import argparse

import tensorflow as tf
from tensorflow.keras.layers import (Concatenate, Conv1D, Dense, Dropout,
                                     Flatten, Input, Lambda, MaxPooling1D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from src.config import DEFAULT_DATA_DIR
from src.evaluator import ClassifierAdapter, evaluate_run
from src.paths import ALL_STYPES

CLASSIFIER_NAME = "cnn"

# Fixed range for the histogram bucketisation. WISDM watch accelerometer is in m/s²
# and never leaves [-50, 50] in practice; using a fixed range keeps the op
# differentiable-friendly and shape-static for the Keras graph.
_HIST_RANGE = (-50.0, 50.0)
_HIST_BINS = 10


def statistical_features_native(x, bins: int = _HIST_BINS, value_range=_HIST_RANGE):
    """Vectorised per-sample features. Input (B, T, C) → output (B, C*(3+bins))."""
    # mean / variance / sum_abs per channel — straight tf ops, fully batched
    mean    = tf.reduce_mean(x, axis=1)                  # (B, C)
    var     = tf.math.reduce_variance(x, axis=1)         # (B, C)
    sum_abs = tf.reduce_sum(tf.abs(x), axis=1)           # (B, C)

    # Histograms: tf.histogram_fixed_width_bins gives bin indices,
    # then one-hot + reduce_sum to count occurrences per bin. Vectorised
    # across the whole batch.
    bin_ids = tf.histogram_fixed_width_bins(x, value_range=value_range, nbins=bins)  # (B, T, C)
    one_hot = tf.one_hot(bin_ids, depth=bins, dtype=tf.float32)                      # (B, T, C, bins)
    hist = tf.reduce_sum(one_hot, axis=1)                                            # (B, C, bins)

    channels = x.shape[-1]
    hist = tf.reshape(hist, (-1, channels * bins))       # (B, C*bins)
    stats = tf.concat([mean, var, sum_abs], axis=-1)     # (B, 3*C)
    return tf.concat([stats, hist], axis=-1)             # (B, C*(3+bins))


def _build(input_shape, num_classes=6, l2_reg=0.0001, bins=_HIST_BINS):
    inp = Input(shape=input_shape)
    conv = Conv1D(filters=196, kernel_size=16, strides=1, activation="relu",
                  kernel_regularizer=l2(l2_reg))(inp)
    pool = MaxPooling1D(pool_size=4)(conv)
    flat = Flatten()(pool)
    stats = Lambda(lambda t: statistical_features_native(t, bins=bins))(inp)
    concat = Concatenate()([flat, stats])
    fc = Dense(1024, activation="relu", kernel_regularizer=l2(l2_reg))(concat)
    fc = Dropout(0.05)(fc)
    out = Dense(num_classes, activation="softmax")(fc)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _adapter(epochs: int, batch_size: int) -> ClassifierAdapter:
    return ClassifierAdapter(
        name=CLASSIFIER_NAME,
        build=lambda shape: _build(shape),
        fit=lambda m, X, y: (m.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2), m)[1],
        predict_labels=lambda m, X: m.predict(X, verbose=0).argmax(axis=-1),
        save=lambda m, p: m.save(p),
        load=lambda p: load_model(p, custom_objects={
            "statistical_features_native": statistical_features_native,
        }),
        flatten_for_model=False,
    )


def run(stype, sampling="loto", fold=0, n_folds=3, dataset_folder=DEFAULT_DATA_DIR,
        force=False, train_new=False, epochs=20, batch_size=64):
    return evaluate_run(
        adapter=_adapter(epochs, batch_size), stype=stype, sampling=sampling,
        fold=fold, n_folds=n_folds, dataset_folder=dataset_folder,
        force=force, train_new=train_new,
        extra={"epochs": epochs, "batch_size": batch_size, "hist_bins": _HIST_BINS},
    )


def build_argparser(prog=None):
    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stype", default="GAF", choices=ALL_STYPES)
    p.add_argument("--sampling", default="loto", choices=["loto", "loso"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--data-folder", default=DEFAULT_DATA_DIR)
    p.add_argument("--force", action="store_true")
    p.add_argument("--train-new", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    return p


def main():
    a = build_argparser().parse_args()
    run(stype=a.stype, sampling=a.sampling, fold=a.fold, n_folds=a.n_folds,
        dataset_folder=a.data_folder, force=a.force, train_new=a.train_new,
        epochs=a.epochs, batch_size=a.batch_size)


if __name__ == "__main__":
    main()
