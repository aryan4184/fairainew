
import os
import sys
import logging
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.optim as optim

import fairlearn.metrics as fair_metrics

from tqdm import tqdm


# Reduce matplotlib noise
matplotlib.set_loglevel("warning")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset_defer.broward import BrowardDataset
from networks.linear_net import LinearNet
from methods.faircomb import PL_Combine_Fair

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    logging.info(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    logging.info("Using device: CPU")

basedir = os.path.dirname(__file__)
DATA_DIR = os.path.join(basedir, "../data")
PLOT_DIR = os.path.join(basedir, "../exp_data/plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# Metric helper
def compute_eod_metrics(data, class_num=2):
    """
    Compute Equalized Odds Difference per class
    """
    res = {}
    labels = data["labels"]
    demographics = data["demographics"]
    combined_preds = np.array(data["combined_preds"])

    for c in range(class_num):
        res[f"system_equalized_odds_difference_c{c}"] = (
            fair_metrics.equalized_odds_difference(
                labels == c,
                combined_preds == c,
                sensitive_features=demographics,
            )
        )
    return res


# Main experiment
def main():
    # Hyperparameters
    K_VALUES = [2, 5, 10, 20, 30]
    THRESHOLD_RANGE = np.concatenate((np.arange(0.0, 2.0, 0.2), np.arange(2.0, 20.0, 2.0), np.arange(20.0, 101.0, 10.0)))
    TOTAL_EPOCHS = 50 
    LR = 1e-2

    # Load dataset once
    dataset = BrowardDataset(DATA_DIR)

    results = {}

    for K in K_VALUES:
        logging.info(f"Running experiment for K = {K}")
        eqd_data = []

        # Model + method
        # Dataset has 2 classes
        model = LinearNet(dataset.d, 2).to(device)
        pl_fair = PL_Combine_Fair(model, device, k=K)

        pl_fair.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=TOTAL_EPOCHS,
            optimizer=optim.Adam,
            lr=LR,
            verbose=False,
            test_interval=5,
        )

        # Sweep fairness threshold
        for r in THRESHOLD_RANGE:
            output = pl_fair.test(dataset.data_test_loader, fairness_cost=r)
            metrics_dict = compute_eod_metrics(output, class_num=2)

            eqd_data.append(
                {
                    "threshold": r,
                    "EQD_c0": metrics_dict.get("system_equalized_odds_difference_c0", 0),
                    "EQD_c1": metrics_dict.get("system_equalized_odds_difference_c1", 0),
                    "deferral_rate": np.mean(output["defers"]),
                }
            )

            logging.info(f"K={K}, r={r:.2f} done.")

        results[K] = pd.DataFrame(eqd_data)

    # Plot results

    for K, df in results.items():
        plt.figure(figsize=(10, 6))
        plt.plot(df["threshold"], df["EQD_c0"], label="Class 0")
        plt.plot(df["threshold"], df["EQD_c1"], label="Class 1")
        plt.plot(df["threshold"], df["deferral_rate"], label="Deferral Rate")

        plt.title(f"COMPAS EOD and Deferral Rate vs Threshold (K={K})")
        plt.xlabel("Fairness Cost (r)")
        plt.ylabel("EOD / Deferral Rate")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(PLOT_DIR, f"compass_eqd_k{K}.png")
        plt.savefig(plot_path)
        plt.close()

        # Save CSV
        csv_path = os.path.join(PLOT_DIR, f"compass_results_k{K}.csv")
        df.to_csv(csv_path, index=False)

        logging.info(f"Saved plots and CSV for K={K}")

    logging.info("All experiments completed successfully.")


if __name__ == "__main__":
    main()
