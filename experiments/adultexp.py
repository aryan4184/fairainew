
import os
import sys
import logging

import numpy as np
import torch
import torch.optim as optim
import fairlearn.metrics as metrics

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset_defer.AdultDataset import Adult
from networks.linear_net import LinearNet
from methods.costcombination import PL_Combine_Cost
from methods.combination import PL_Combine
from methods.faircomb import PL_Combine_Fair
from helpers.metrics import accuracy_gap_per_group

import random
import numpy as np

logging.basicConfig(level=logging.DEBUG)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def combine_defer(preds, h_preds, defers):
    """Combine model and human predictions using deferral decisions"""
    return preds * (1 - defers) + h_preds * defers


def print_metrics(data, class_num=2, combine_method="defer"):
    """
    Compute and print fairness + accuracy metrics
    for each class and demographic group
    """
    res = {}


    for positive_class in range(class_num):
        preds = (data["preds"] == positive_class)
        labels = (data["labels"] == positive_class)
        hpreds = (data["hum_preds"] == positive_class)
        defers = data["defers"]
        demographics = data["demographics"]

        if combine_method == "defer":
            combined_preds = combine_defer(preds, hpreds, defers)
        else:  # PL
            combined_preds = (np.array(data["combined_preds"]) == positive_class)

        for demographic in set(demographics):
            mask = demographics == demographic

            tp = ((combined_preds[mask] == 1) & (labels[mask] == 1)).sum()
            tn = ((combined_preds[mask] == 0) & (labels[mask] == 0)).sum()
            fp = ((combined_preds[mask] == 1) & (labels[mask] == 0)).sum()
            fn = ((combined_preds[mask] == 0) & (labels[mask] == 1)).sum()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            res[f"Class {positive_class} | Demographic {demographic} | TPR"] = tpr
            res[f"Class {positive_class} | Demographic {demographic} | FPR"] = fpr
            res[f"Class {positive_class} | Demographic {demographic} | TNR"] = tnr
            res[f"Class {positive_class} | Demographic {demographic} | FNR"] = fnr

        # Fairness metrics
        res[f"model_demographic_parity_difference_c{positive_class}"] = (
            metrics.demographic_parity_difference(labels, preds, sensitive_features=demographics)
        )
        res[f"human_demographic_parity_difference_c{positive_class}"] = (
            metrics.demographic_parity_difference(labels, hpreds, sensitive_features=demographics)
        )
        res[f"system_demographic_parity_c{positive_class}"] = (
            metrics.demographic_parity_difference(labels, combined_preds, sensitive_features=demographics)
        )


        res[f"model_equalized_odds_difference_c{positive_class}"] = (
            metrics.equalized_odds_difference(labels, preds, sensitive_features=demographics)
        )
        res[f"human_equalized_odds_difference_c{positive_class}"] = (
            metrics.equalized_odds_difference(labels, hpreds, sensitive_features=demographics)
        )
        res[f"system_equalized_odds_difference_c{positive_class}"] = (
            metrics.equalized_odds_difference(labels, combined_preds, sensitive_features=demographics)
        )

    # Accuracy + deferral
    res["deferral rate"] = data["defers"].mean()
    res["model accuracy"] = (data["preds"] == data["labels"]).mean()
    res["human accuracy"] = (data["hum_preds"] == data["labels"]).mean()

    if combine_method == "defer":
        combined_preds = combine_defer(data["preds"], data["hum_preds"], data["defers"])
    else:
        combined_preds = np.array(data["combined_preds"])

    res["combined accuracy"] = (combined_preds == data["labels"]).mean()
   
    # Accuracy gap per demographic
    acc_gap_res = accuracy_gap_per_group(
        data,
        use_combined=(combine_method == "defer")
    )

    res.update(acc_gap_res)


    for k, v in res.items():
        print(k, ":", v)

    return res


def summarize_metrics(trial_results):
    """Print mean and variance over multiple trials"""
    if not trial_results:
        print("No trial results to summarize.")
        return

    for key in trial_results[0]:
        values = [trial[key] for trial in trial_results]
        print(f"{key}: mean={np.mean(values)}, var={np.var(values)}")


# Main experiment loop 
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    args = parser.parse_args()

    # Output directories
    basedir = os.path.dirname(__file__)
    base_dir = os.path.join(basedir, "../exp_data")
    for sub in ["data", "plots", "models"]:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam
    scheduler = None
    lr = args.lr

    data_dir = os.path.join(basedir, "../data")
    max_trials = args.trials
    total_epochs = args.epochs

    stats_cost = []
    stats_all = []
    stats_fair = []

    for trial in range(max_trials):
        set_seed(trial + 42)
        dataset = Adult(data_dir, device, random_seed=trial + 42)

        # Cost-optimized combination
        model = LinearNet(dataset.d, 4).to(device)
        plc = PL_Combine_Cost(model, device)
        plc.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
        )
        torch.save(plc.model_class.state_dict(), os.path.join(base_dir, "models", f"linear_cost_trial{trial}.pt"))
        output = plc.test(dataset.data_test_loader)
        print("\nFairness Metrics (Cost Optimized):")
        stats_cost.append(print_metrics(output, class_num=3, combine_method="PL"))


        # All combination
        model = LinearNet(dataset.d, 4).to(device)
        plc = PL_Combine(model, device)
        plc.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
        )
        torch.save(plc.model_class.state_dict(), os.path.join(base_dir, "models", f"linear_all_trial{trial}.pt"))
        output = plc.test(dataset.data_test_loader)
        print("\nFairness Metrics (All):")
        stats_all.append(print_metrics(output, class_num=3, combine_method="PL"))

        # Fair combination
        model = LinearNet(dataset.d, 4).to(device)
        plc = PL_Combine_Fair(model, device, fairness_cost=20.0, human_cost=1.0)
        plc.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
        )
        torch.save(plc.model_class.state_dict(), os.path.join(base_dir, "models", f"linear_fair_trial{trial}.pt"))
        output = plc.test(dataset.data_test_loader)
        print("\nFairness Metrics (Fair):")
        stats_fair.append(print_metrics(output, class_num=3, combine_method="PL"))

    print("\n--- Cost optimized ---")
    summarize_metrics(stats_cost)

    print("\n--- All combination ---")
    summarize_metrics(stats_all)

    print("\n--- Fair combination ---")
    summarize_metrics(stats_fair)

if __name__ == "__main__":
    main()