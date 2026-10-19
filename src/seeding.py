"""Global RNG seeding for reproducible classifier training.

Call ``set_global_seeds()`` at the start of a validator's ``run()`` — before any
model is built or fit — to seed Python, NumPy and TensorFlow. This fixes weight
initialisation, batch shuffling and dropout, which is what makes the reported
neural-network results reproducible.

Note: exact bit-for-bit GPU reproducibility would additionally require
``tf.config.experimental.enable_op_determinism()``, which we deliberately leave
off to avoid the throughput hit — seeding the RNGs is the proportionate fix here.
KNN and SVM are deterministic (no RNG in fit); Random Forest passes this seed via
``random_state``.
"""

from __future__ import annotations

# Single source of truth for the seed cited in the paper.
DEFAULT_SEED = 95


def set_global_seeds(seed: int = DEFAULT_SEED) -> None:
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
