"""
evaluate.py
-----------
Objective
    Final evaluation of the trained network on the held-out TEST set
    (never seen during training or validation), and produce the analysis
    artifacts required by the assignment:
        - overall test accuracy
        - per-class precision / recall / F1 (computed manually with NumPy)
        - a 35x35 confusion matrix (computed manually, plotted as a heatmap)
        - at least 5 misclassified examples, saved as images with their
          true/predicted labels for qualitative error analysis

Outputs
    outputs/plots/confusion_matrix.png
    outputs/plots/misclassified_examples.png
    outputs/test_metrics.txt
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
WEIGHTS_PATH = "outputs/weights/model.npz"
CM_PLOT_OUT = "outputs/plots/confusion_matrix.png"
MISCLASSIFIED_PLOT_OUT = "outputs/plots/misclassified_examples.png"
METRICS_TXT_OUT = "outputs/test_metrics.txt"

HIDDEN_LAYERS = [128, 64]
ACTIVATION = "relu"
IMG_SIZE = 28
NUM_MISCLASSIFIED_TO_SHOW = 8  # >= 5 required by the assignment


def confusion_matrix(y_true, y_pred, num_classes):
    """
    Build a confusion matrix from scratch using NumPy.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (n,)
        Integer ground-truth and predicted class labels.
    num_classes : int

    Returns
    -------
    np.ndarray, shape (num_classes, num_classes)
        cm[i, j] = number of samples whose TRUE class is i and that were
        PREDICTED as class j. The diagonal holds correct predictions.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm):
    """
    Compute per-class precision, recall, and F1 score directly from a
    confusion matrix, plus their macro averages.

    Definitions
    -----------
    For class i:
        TP = cm[i, i]
        FP = sum of column i, excluding the diagonal entry
        FN = sum of row i, excluding the diagonal entry
        Precision = TP / (TP + FP)   -> "of predicted class i, how many were right"
        Recall    = TP / (TP + FN)   -> "of actual class i, how many were found"
        F1        = harmonic mean of precision and recall

    Parameters
    ----------
    cm : np.ndarray, shape (num_classes, num_classes)

    Returns
    -------
    precision, recall, f1 : np.ndarray, shape (num_classes,)
    macro_precision, macro_recall, macro_f1 : float
    """
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    eps = 1e-9

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp + eps)
        recall[i] = tp / (tp + fn + eps)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + eps)

    return precision, recall, f1, precision.mean(), recall.mean(), f1.mean()


def plot_confusion_matrix(cm, classes, out_path):
    """Render the confusion matrix as a heatmap and save it to disk."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - Test Set (35 classes)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_misclassified(X, y_true, y_pred, classes, indices, out_path):
    """
    Plot a grid of misclassified test images with their true vs.
    predicted labels for qualitative analysis.

    Parameters
    ----------
    X : np.ndarray, shape (n, IMG_SIZE*IMG_SIZE)
        Test images (flattened).
    y_true, y_pred : np.ndarray, shape (n,)
        True and predicted integer labels for those images.
    classes : list[str]
        Index -> character name mapping.
    indices : list[int]
        Which rows of X/y_true/y_pred to plot (the misclassified ones).
    out_path : str
    """
    n = len(indices)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_idx, sample_idx in enumerate(indices):
        ax = axes[ax_idx]
        img = X[sample_idx].reshape(IMG_SIZE, IMG_SIZE)
        ax.imshow(img, cmap="gray")
        true_c = classes[y_true[sample_idx]]
        pred_c = classes[y_pred[sample_idx]]
        ax.set_title(f"True: {true_c}  Pred: {pred_c}", fontsize=11, color="crimson")
        ax.axis("off")

    for ax_idx in range(n, len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle("Misclassified Test Examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    """Run full test-set evaluation and error analysis."""
    data = np.load(SPLITS_PATH, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]
    classes = list(data["classes"])
    num_classes = len(classes)
    input_dim = X_test.shape[1]

    layer_sizes = [input_dim] + HIDDEN_LAYERS + [num_classes]
    net = NeuralNetwork(layer_sizes=layer_sizes, activation=ACTIVATION)
    net.load(WEIGHTS_PATH)

    y_pred = net.predict(X_test)
    test_acc = float(np.mean(y_pred == y_test))

    cm = confusion_matrix(y_test, y_pred, num_classes)
    precision, recall, f1, macro_p, macro_r, macro_f1 = precision_recall_f1(cm)

    plot_confusion_matrix(cm, classes, CM_PLOT_OUT)

    # ---- Misclassification analysis ----
    misclassified_idx = np.where(y_pred != y_test)[0]
    print(f"[evaluate] Total misclassified test samples: {len(misclassified_idx)} / {len(y_test)}")
    rng = np.random.default_rng(0)
    chosen = rng.choice(
        misclassified_idx,
        size=min(NUM_MISCLASSIFIED_TO_SHOW, len(misclassified_idx)),
        replace=False,
    )
    plot_misclassified(X_test, y_test, y_pred, classes, chosen, MISCLASSIFIED_PLOT_OUT)

    # ---- Per-class report ----
    lines = []
    lines.append(f"TEST SET ACCURACY: {test_acc:.4f}  ({np.sum(y_pred==y_test)}/{len(y_test)} correct)\n")
    lines.append(f"Macro Precision: {macro_p:.4f} | Macro Recall: {macro_r:.4f} | Macro F1: {macro_f1:.4f}\n")
    lines.append(f"{'Class':<8}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    for i, c in enumerate(classes):
        lines.append(f"{c:<8}{precision[i]:<12.3f}{recall[i]:<12.3f}{f1[i]:<12.3f}")

    lines.append("\nMisclassified sample details (true -> predicted):")
    for idx in chosen:
        lines.append(f"  test_idx={idx}: true={classes[y_test[idx]]} -> pred={classes[y_pred[idx]]}")

    report_text = "\n".join(lines)
    with open(METRICS_TXT_OUT, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n[evaluate] Saved confusion matrix -> {CM_PLOT_OUT}")
    print(f"[evaluate] Saved misclassified examples -> {MISCLASSIFIED_PLOT_OUT}")
    print(f"[evaluate] Saved full metrics report -> {METRICS_TXT_OUT}")

    return chosen, classes, y_test, y_pred


if __name__ == "__main__":
    main()
