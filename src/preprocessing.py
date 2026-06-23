"""
preprocessing.py
----------------
Objective
    Take the raw generated dataset (data/raw_dataset.npz) and produce a
    clean train / validation / test split that the neural network can
    consume directly.

What this module does
    1. Loads the flattened, normalized image vectors and integer labels.
    2. Performs a STRATIFIED split (each class contributes the same
       proportion to train/val/test) so that no split is missing or
       under-representing any of the 35 classes.
    3. One-hot encodes the labels for use with the cross-entropy loss /
       softmax output layer.
    4. Saves the final splits to data/splits.npz for the training script.

Split ratio: 70% train / 15% validation / 15% test.
"""

import numpy as np
import os

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer class labels into one-hot encoded vectors.

    Parameters
    ----------
    labels : np.ndarray, shape (N,)
        Integer class indices in [0, num_classes).
    num_classes : int
        Total number of classes (35 here).

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        One-hot encoded label matrix, required by the softmax +
        cross-entropy output layer of the network.
    """
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def stratified_split(images: np.ndarray, labels: np.ndarray, num_classes: int,
                      train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                      seed=RANDOM_SEED):
    """
    Perform a class-stratified train/validation/test split.

    Stratification matters here because each of the 35 classes was
    generated with exactly the same number of samples; a stratified split
    keeps that balance in every subset, which keeps accuracy/F1 metrics
    meaningful and prevents any split from accidentally missing a class.

    Parameters
    ----------
    images : np.ndarray, shape (N, D)
        Flattened, normalized image vectors.
    labels : np.ndarray, shape (N,)
        Integer class labels.
    num_classes : int
        Number of distinct classes.
    train_ratio, val_ratio : float
        Proportions for train and validation; test gets the remainder.
    seed : int
        Seed for reproducible shuffling.

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
        (y_* are integer labels; one-hot encoding happens separately)
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n = len(idx_c)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(idx_c[:n_train])
        val_idx.extend(idx_c[n_train:n_train + n_val])
        test_idx.extend(idx_c[n_train + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return {
        "X_train": images[train_idx], "y_train": labels[train_idx],
        "X_val": images[val_idx], "y_val": labels[val_idx],
        "X_test": images[test_idx], "y_test": labels[test_idx],
    }


def prepare_and_save(raw_path="data/raw_dataset.npz", out_path="data/splits.npz"):
    """
    Full preprocessing pipeline: load raw data, verify normalization,
    split it, one-hot encode labels, and persist everything to disk.

    Parameters
    ----------
    raw_path : str
        Path to the .npz file produced by data_generation.py.
    out_path : str
        Where to write the final splits.

    Returns
    -------
    None (writes data/splits.npz)
    """
    data = np.load(raw_path, allow_pickle=True)
    images, labels, classes = data["images"], data["labels"], data["classes"]
    num_classes = len(classes)

    # Sanity check: pixel values must already be normalized to [0, 1]
    assert images.min() >= 0.0 and images.max() <= 1.0, \
        "Images must be normalized to [0, 1] before training."

    splits = stratified_split(images, labels, num_classes)

    y_train_oh = one_hot(splits["y_train"], num_classes)
    y_val_oh = one_hot(splits["y_val"], num_classes)
    y_test_oh = one_hot(splits["y_test"], num_classes)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        X_train=splits["X_train"], y_train=splits["y_train"], y_train_oh=y_train_oh,
        X_val=splits["X_val"], y_val=splits["y_val"], y_val_oh=y_val_oh,
        X_test=splits["X_test"], y_test=splits["y_test"], y_test_oh=y_test_oh,
        classes=classes,
    )

    print("[preprocessing] Split sizes:")
    print(f"  Train: {splits['X_train'].shape[0]} samples")
    print(f"  Val:   {splits['X_val'].shape[0]} samples")
    print(f"  Test:  {splits['X_test'].shape[0]} samples")
    print(f"[preprocessing] Saved -> {out_path}")


if __name__ == "__main__":
    prepare_and_save()
