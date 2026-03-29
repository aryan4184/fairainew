import os
import sys
import pickle
import argparse
import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

import fairlearn.metrics as metrics
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Enable imports from the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Core model definition
from networks.linear_net import *

# Dataset used for experiments
from dataset_defer.hatespeech import *

# Threshold-based utilities (used by some methods)
from methods.seperate_thresholds import *

# Learning-to-defer combination methods
from methods.costcombination import *
from methods.combination import *
from methods.faircomb import *

# Configure logging to show all debug information
logging.basicConfig(level=logging.DEBUG)


def combine_defer(preds, h_preds, defers):
    """
    Combine model predictions and human predictions using a deferral mask.
    If defers == 1, use human prediction; otherwise use model prediction.
    """
    return preds * (1 - defers) + h_preds * defers


def print_metrics(data, class_num=3, combine_method="defer"):
    """
    Compute and print fairness and performance metrics for each class and demographic group.

    Parameters:
        data (dict): Output dictionary from a model's test() method
        class_num (int): Number of prediction classes
        combine_method (str): Either 'defer' or 'PL' for combination strategy

    Returns:
        dict: Dictionary of computed metrics
    """
    res = dict()

    for positive_class in range(class_num):
        preds = (data["preds"] == positive_class)
        labels = (data["labels"] == positive_class)
        hpreds = (data["hum_preds"] == positive_class)
        defers = data["defers"]
        demographics = data["demographics"]

        if combine_method == "defer":
            combined_preds = combine_defer(preds, hpreds, defers)
        elif combine_method == "PL":
            combined_preds = np.array(data["combined_preds"])
            combined_preds = (combined_preds == positive_class)

        for demographic in set(demographics):
            mask = (demographics == demographic)
            masked_labels = labels[mask]
            masked_preds = combined_preds[mask]

            tp = ((masked_preds == 1) & (masked_labels == 1)).sum()
            tn = ((masked_preds == 0) & (masked_labels == 0)).sum()
            fp = ((masked_preds == 1) & (masked_labels == 0)).sum()
            fn = ((masked_preds == 0) & (masked_labels == 1)).sum()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            res[f"Class {positive_class} | Demographic {demographic} | TPR"] = tpr
            res[f"Class {positive_class} | Demographic {demographic} | FPR"] = fpr
            res[f"Class {positive_class} | Demographic {demographic} | TNR"] = tnr
            res[f"Class {positive_class} | Demographic {demographic} | FNR"] = fnr

        res[f"model_demographic_parity_difference_c{positive_class}"] = \
            metrics.demographic_parity_difference(labels, preds, sensitive_features=demographics)

        res[f"human_demographic_parity_difference_c{positive_class}"] = \
            metrics.demographic_parity_difference(labels, hpreds, sensitive_features=demographics)

        res[f"system_demographic_parity_c{positive_class}"] = \
            metrics.demographic_parity_difference(labels, combined_preds, sensitive_features=demographics)

        res[f"model_equalized_odds_difference_c{positive_class}"] = \
            metrics.equalized_odds_difference(labels, preds, sensitive_features=demographics)

        res[f"human_equalized_odds_difference_c{positive_class}"] = \
            metrics.equalized_odds_difference(labels, hpreds, sensitive_features=demographics)

        res[f"system_equalized_odds_difference_c{positive_class}"] = \
            metrics.equalized_odds_difference(labels, combined_preds, sensitive_features=demographics)

    res["deferral rate"] = data["defers"].mean()
    res["model accuracy"] = (data["preds"] == data["labels"]).mean()
    res["human accuracy"] = (data["hum_preds"] == data["labels"]).mean()

    if combine_method == "defer":
        combined_preds = combine_defer(preds, hpreds, defers)
    elif combine_method == "PL":
        combined_preds = np.array(data["combined_preds"])

    res["combined accuracy"] = (combined_preds == data["labels"]).mean()

    for key, value in res.items():
        print(key, ":", value)

    return res


def summarize_metrics(trial_results):
    """
    Aggregate metrics across multiple trials by computing mean and variance.
    """
    if not trial_results:
        print("No trial results to summarize.")
        return

    keys = trial_results[0].keys()

    for key in keys:
        values = [trial[key] for trial in trial_results if key in trial]
        print(f"{key}: {np.mean(values)}")


def store_test_results_to_csv(test_data, csv_path="results.csv"):
    """
    Save selected test-time outputs to a CSV file for post-hoc analysis.
    """
    df = pd.DataFrame({
        "max_probs": test_data["max_probs"],
        "hum_preds": test_data["hum_preds"],
        "demographics": test_data["demographics"],
    })
    df.to_csv(csv_path, index=False)
    print(f"Data successfully written to {csv_path}")


def main():
    """
    Run multiple trials comparing different learning-to-defer strategies
    on the Hate Speech dataset and report fairness metrics.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    args = parser.parse_args()

    basedir = os.path.dirname(__file__)
    base_dir = os.path.join(basedir, "../exp_data")
    for subdir in ["", "data", "plots", "models"]:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(basedir, "../data")

    optimizer = optim.Adam
    scheduler = None
    lr = args.lr
    max_trials = args.trials
    total_epochs = args.epochs

    stats_combination_cost = []
    stats_combination_all = []
    stats_combination_fair = []

    for trial in range(max_trials):
        set_seed(trial + 42)
        # dataset = HateSpeech(data_dir, True, False, "random_annotator", device)
        dataset = HateSpeech(data_dir, True, False, "synthetic", device, synth_exp_param=[0.95, 0.92])

        model = LinearNet(dataset.d, 4).to(device)
        PLC = PL_Combine_Cost(model, device, miss_cost=20.0, human_cost=1.0)
        PLC.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )
        torch.save(PLC.model_class.state_dict(), os.path.join(base_dir, "models", f"hatespeech_cost_trial{trial}.pt"))
        output = PLC.test(dataset.data_test_loader)
        print("\n\nFairness Metrics for cost optimized combination:")
        stats_combination_cost.append(print_metrics(output, class_num=3, combine_method="PL"))

        model = LinearNet(dataset.d, 4).to(device)
        PLC = PL_Combine(model, device)
        PLC.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )
        torch.save(PLC.model_class.state_dict(), os.path.join(base_dir, "models", f"hatespeech_all_trial{trial}.pt"))
        output = PLC.test(dataset.data_test_loader)
        print("\n\nFairness Metrics for all combination:")
        stats_combination_all.append(print_metrics(output, class_num=3, combine_method="PL"))

        model = LinearNet(dataset.d, 4).to(device)
        PLC = PL_Combine_Fair(model, device, fairness_cost=20.0, human_cost=1.0)
        PLC.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )
        torch.save(PLC.model_class.state_dict(), os.path.join(base_dir, "models", f"hatespeech_fair_trial{trial}.pt"))
        output = PLC.test(dataset.data_test_loader)
        print("\n\nFairness Metrics for fair combination:")
        stats_combination_fair.append(print_metrics(output, class_num=3, combine_method="PL"))

    print("\n\n-- Stats of cost optimized combination")
    summarize_metrics(stats_combination_cost)

    print("\n\n-- Stats of all combination")
    summarize_metrics(stats_combination_all)

    print("\n\n-- Stats of fair combination")
    summarize_metrics(stats_combination_fair)


if __name__ == "__main__":
    main()
