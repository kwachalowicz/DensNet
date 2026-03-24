import argparse
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import tensorflow as tf


def load_test_data(test_x_path, test_y_path, test_samples=None):
    with h5py.File(test_x_path, 'r') as fx:
        x_test = fx['x'][:].astype('float32') / 255.0

    with h5py.File(test_y_path, 'r') as fy:
        y_test = np.squeeze(fy['y'][:]).astype('float32')

    if test_samples and test_samples > 0:
        x_test = x_test[:test_samples]
        y_test = y_test[:test_samples]

    return x_test, y_test


def evaluate_model(model, x_test, y_test, batch_size=16):
    y_pred_proba = model.predict(x_test, batch_size=batch_size, verbose=1)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    y_pred_proba = y_pred_proba.flatten()

    y_test_flat = y_test.flatten() if len(y_test.shape) > 1 else y_test

    accuracy = accuracy_score(y_test_flat, y_pred)
    precision = precision_score(y_test_flat, y_pred, zero_division=0)
    recall = recall_score(y_test_flat, y_pred, zero_division=0)
    f1 = f1_score(y_test_flat, y_pred, zero_division=0)
    auc = roc_auc_score(y_test_flat, y_pred_proba)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    return metrics, y_pred, y_pred_proba, y_test_flat


def save_metrics_report(metrics, y_test, y_pred, output_dir='.'):
    report_path = os.path.join(output_dir, "evaluation_report.txt")

    with open(report_path, "w") as f:
        f.write("MODEL EVALUATION REPORT\n\n")
        f.write("SUMMARY METRICS\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC:   {metrics['auc']:.4f}\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write(classification_report(
            y_test,
            y_pred,
            target_names=["Negative (0)", "Positive (1)"]
        ))


def plot_roc_curve(y_test, y_pred_proba, output_dir='.'):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_test, y_pred, output_dir='.'):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_metrics_bar(metrics, output_dir='.'):
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values)

    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    plt.ylim([0, 1])

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom"
        )

    metrics_path = os.path.join(output_dir, "metrics_chart.png")
    plt.savefig(metrics_path, dpi=100, bbox_inches="tight")
    plt.close()


def main(args):
    x_test, y_test = load_test_data(args.test_x, args.test_y, args.test_samples)

    model = tf.keras.models.load_model(args.model)

    metrics, y_pred, y_pred_proba, y_test_flat = evaluate_model(
        model,
        x_test,
        y_test,
        batch_size=args.batch_size
    )

    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)

    save_metrics_report(metrics, y_test_flat, y_pred, output_dir)
    plot_roc_curve(y_test_flat, y_pred_proba, output_dir)
    plot_confusion_matrix(y_test_flat, y_pred, output_dir)
    plot_metrics_bar(metrics, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--test-x", required=True)
    parser.add_argument("--test-y", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    main(args)