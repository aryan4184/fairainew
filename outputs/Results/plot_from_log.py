import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_epoch_metrics(log_path):
    """
    Parse lines like:
    INFO:root:EPOCH_METRICS epoch=0 model_acc=0.6475 human_acc=0.6455 combined_acc=0.6455 eod_c0=0.1 eod_c1=0.2 eod_c2=0.3
    Keeps only the last metrics for each epoch.
    """
    epoch_metrics = {}

    # Improved float regex to accept numbers like 0.6475 and handle nan properly
    float_re = r"(\d+(?:\.\d+)?)"
    nan_re = r"(nan)"

    pattern = re.compile(
        rf"epoch=(\d+)\s+model_acc={float_re}\s+human_acc={float_re}\s+combined_acc={float_re}"
        rf"(?:\s+eod_c0=({float_re}|{nan_re}))?"
        rf"(?:\s+eod_c1=({float_re}|{nan_re}))?"
        rf"(?:\s+eod_c2=({float_re}|{nan_re}))?"
    )

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                # Groups:
                # 1: epoch
                # 2: model_acc
                # 3: human_acc
                # 4: combined_acc
                # 5: eod_c0
                # 6: eod_c1
                # 7: eod_c2

                def to_float_or_nan(s):
                    return float(s) if s != "nan" else np.nan

                epoch_metrics[epoch] = {
                    "model_acc": float(match.group(2)),
                    "human_acc": float(match.group(3)),
                    "combined_acc": float(match.group(4)),
                    "eod_c0": to_float_or_nan(match.group(5)) if match.group(5) else np.nan,
                    "eod_c1": to_float_or_nan(match.group(6)) if match.group(6) else np.nan,
                    "eod_c2": to_float_or_nan(match.group(7)) if match.group(7) else np.nan,
                }

    # Sort epochs
    epochs = sorted(epoch_metrics.keys())

    # Build lists for plotting
    result = {
        "epoch": epochs,
        "model_acc": [epoch_metrics[e]["model_acc"] for e in epochs],
        "human_acc": [epoch_metrics[e]["human_acc"] for e in epochs],
        "combined_acc": [epoch_metrics[e]["combined_acc"] for e in epochs],
        "eod_c0": [epoch_metrics[e]["eod_c0"] for e in epochs],
        "eod_c1": [epoch_metrics[e]["eod_c1"] for e in epochs],
        "eod_c2": [epoch_metrics[e]["eod_c2"] for e in epochs],
    }
    return result


def plot_epoch_line(epoch_metrics, metric_name, ylabel, title):
    epochs = epoch_metrics["epoch"]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_metrics[f"model_{metric_name}"], label="Model", marker='o', linestyle='-')
    plt.plot(epochs, epoch_metrics[f"human_{metric_name}"], label="Human", marker='s', linestyle='--')
    plt.plot(epochs, epoch_metrics[f"combined_{metric_name}"], label="Combined", marker='^', linestyle=':')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_epoch_eod(epoch_metrics, cls):
    epochs = epoch_metrics["epoch"]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_metrics[f"eod_c{cls}"], label=f"Class {cls} EOD", marker='o', color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("Equalized Odds Difference")
    plt.title(f"EOD over Epochs (Class {cls})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    args = parser.parse_args()

    epoch_metrics = parse_epoch_metrics(args.log)

    # Debug print to check parsed values
    print("Parsed Human Accuracy:", epoch_metrics["human_acc"])

    # Plot Accuracy over epochs
    plot_epoch_line(epoch_metrics, "acc", "Accuracy", "Accuracy over Epochs")

    # Plot EOD for each class over epochs
    for cls in range(3):
        plot_epoch_eod(epoch_metrics, cls)


if __name__ == "__main__":
    main()
