"""
train.py
--------
Objective
    Glue script that loads the preprocessed train/val/test splits, builds
    the from-scratch NeuralNetwork, trains it, and saves:
        - trained weights              -> outputs/weights/model.npz
        - loss/accuracy curve plot     -> outputs/plots/training_curves.png
        - a metrics summary text file  -> outputs/metrics_summary.txt

Run this after data_generation.py and preprocessing.py have both been run.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from neural_network import NeuralNetwork

SPLITS_PATH = "data/splits.npz"
WEIGHTS_OUT = "outputs/weights/model.npz"
PLOT_OUT = "outputs/plots/training_curves.png"
METRICS_OUT = "outputs/metrics_summary.txt"

# Hyperparameters (documented here for the report)
HIDDEN_LAYERS = [128, 64]
ACTIVATION = "relu"
LEARNING_RATE = 0.5
L2_LAMBDA = 1e-4
EPOCHS = 60
BATCH_SIZE = 64


def plot_curves(history, out_path):
    """
    Plot training/validation loss and accuracy curves side by side and
    save the figure to disk.

    Parameters
    ----------
    history : dict
        Output of NeuralNetwork.train(), containing train_loss, train_acc,
        val_loss, val_acc lists (one value per epoch).
    out_path : str
        File path to save the resulting PNG plot.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    """Run the full training pipeline end to end."""
    data = np.load(SPLITS_PATH, allow_pickle=True)
    X_train, y_train_int, y_train_oh = data["X_train"], data["y_train"], data["y_train_oh"]
    X_val, y_val_int, y_val_oh = data["X_val"], data["y_val"], data["y_val_oh"]
    classes = data["classes"]
    num_classes = len(classes)
    input_dim = X_train.shape[1]

    layer_sizes = [input_dim] + HIDDEN_LAYERS + [num_classes]
    print(f"[train] Network architecture: {layer_sizes}, activation={ACTIVATION}")

    net = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation=ACTIVATION,
        learning_rate=LEARNING_RATE,
        l2_lambda=L2_LAMBDA,
        seed=42,
    )

    history = net.train(
        X_train, y_train_oh, y_train_int,
        X_val, y_val_oh, y_val_int,
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True,
    )

    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_OUT), exist_ok=True)
    net.save(WEIGHTS_OUT)
    plot_curves(history, PLOT_OUT)

    final_train_acc = history["train_acc"][-1]
    final_val_acc = history["val_acc"][-1]
    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]

    summary = (
        f"Architecture: {layer_sizes}\n"
        f"Activation: {ACTIVATION}\n"
        f"Learning rate: {LEARNING_RATE}\n"
        f"L2 lambda: {L2_LAMBDA}\n"
        f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}\n"
        f"Final Train Loss: {final_train_loss:.4f} | Final Train Acc: {final_train_acc:.4f}\n"
        f"Final Val Loss:   {final_val_loss:.4f} | Final Val Acc:   {final_val_acc:.4f}\n"
    )
    with open(METRICS_OUT, "w") as f:
        f.write(summary)
    print("\n[train] " + summary.replace("\n", "\n[train] "))
    print(f"[train] Saved weights -> {WEIGHTS_OUT}")
    print(f"[train] Saved training curves -> {PLOT_OUT}")


if __name__ == "__main__":
    main()
