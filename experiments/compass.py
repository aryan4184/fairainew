import os
import sys
import logging
import datetime
import numpy as np

import torch
import torch.optim as optim
import fairlearn.metrics as metrics

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from networks.linear_net import *
from methods.costcombination import *
from methods.combination import *
from methods.faircomb import *

from dataset_defer.broward import BrowardDataset

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
    """
    Combine model predictions and human predictions using a deferral mask.
    """
    return preds * (1 - defers) + h_preds * defers


def print_metrics(data, class_num=2, combine_method="defer"):
    """
    Compute and print fairness and performance metrics.
    """
    res = dict()
    
    # Ensure all inputs are numpy or scalar
    # (Handling cases where they might be tensors)
    if torch.is_tensor(data["preds"]): data["preds"] = data["preds"].cpu().numpy()
    if torch.is_tensor(data["labels"]): data["labels"] = data["labels"].cpu().numpy()
    if torch.is_tensor(data["hum_preds"]): data["hum_preds"] = data["hum_preds"].cpu().numpy()
    if torch.is_tensor(data["defers"]): data["defers"] = data["defers"].cpu().numpy()
    if torch.is_tensor(data["demographics"]): data["demographics"] = data["demographics"].cpu().numpy()
    if "combined_preds" in data and torch.is_tensor(data["combined_preds"]):
         data["combined_preds"] = data["combined_preds"].cpu().numpy()

    for positive_class in [1]:
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
            # handle case where no samples for this demo
            if mask.sum() == 0: continue
            
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
        # Ensure correct shape/type logic for combined_preds
        combined_preds = np.array(data["combined_preds"])

    res["combined accuracy"] = (combined_preds == data["labels"]).mean()

    for key, value in res.items():
        print(key, ":", value)

    return res


def summarize_metrics(trial_results):
    """
    Aggregate metrics across multiple trials by computing mean.
    """
    if not trial_results:
        print("No trial results to summarize.")
        return

    keys = trial_results[0].keys()

    print("\n--- Summary Stats ---")
    for key in keys:
        values = [trial[key] for trial in trial_results if key in trial]
        print(f"{key}: {np.mean(values)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    basedir = os.path.dirname(__file__)
    base_dir = os.path.join(basedir, "../exp_data") 
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        os.makedirs(os.path.join(base_dir, "data"))
        os.makedirs(os.path.join(base_dir, "plots"))
        os.makedirs(os.path.join(base_dir, "models"))
    else:
        if not os.path.exists(os.path.join(base_dir, "data")):
            os.makedirs(os.path.join(base_dir, "data"))
        if not os.path.exists(os.path.join(base_dir, "plots")):
            os.makedirs(os.path.join(base_dir, "plots"))
        if not os.path.exists(os.path.join(base_dir, "models")):
            os.makedirs(os.path.join(base_dir, "models"))

    date_now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(basedir, "../data")

    optimizer = optim.AdamW
    scheduler = None
    lr = args.lr
    max_trials = args.trials
    total_epochs = args.epochs

    stats_combination_cost = []
    stats_combination_all = []
    stats_combination_fair = []

    for trial in range(max_trials):
        print(f"Trial {trial+1}/{max_trials}")
        set_seed(trial + 42)
        
        # BrowardDataset uses test_split, val_split
        dataset = BrowardDataset(data_dir, test_split=0.2, val_split=0.1, random_seed=trial + 42)

        # Cost Optimized
        model = LinearNet(dataset.d, 2).to(device)
        PLC = PL_Combine_Cost(model, device)
        PLC.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=20,
        )
        torch.save(PLC.model_class.state_dict(), os.path.join(base_dir, "models", f"compass_cost_trial{trial}.pt"))
        output = PLC.test(dataset.data_test_loader)
        print("\n\nFairness Metrics for cost optimized combination: ")
        stats_combination_cost.append(
            print_metrics(output, class_num=2, combine_method="PL")
        )

        # All Combination
        model = LinearNet(dataset.d, 2).to(device)
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
            test_interval=20,
        )
        torch.save(PLC.model_class.state_dict(), os.path.join(base_dir, "models", f"compass_all_trial{trial}.pt"))
        output = PLC.test(dataset.data_test_loader)
        print("\n\nFairness Metrics for all combination: ")
        stats_combination_all.append(
            print_metrics(output, class_num=2, combine_method="PL")
        )

        # Fair Combination
        model = LinearNet(dataset.d, 2).to(device)
        PLC = PL_Combine_Fair(model, device, fairness_cost=10.0, human_cost=0.1)
        PLC.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=20,
        )
        torch.save(PLC.model_class.state_dict(), os.path.join(base_dir, "models", f"compass_fair_trial{trial}.pt"))
        output = PLC.test(dataset.data_test_loader)
        print("\n\nFairness Metrics for fair combination: ")
        stats_combination_fair.append(
            print_metrics(output, class_num=2, combine_method="PL")
        )

    print("\n\n--Stats of cost optimized unsupervised P+L combiation")
    summarize_metrics(stats_combination_cost)

    print("\n\n--Stats of all combiation")
    summarize_metrics(stats_combination_all)
    
    print("\n\n--Stats of fair combination")
    summarize_metrics(stats_combination_fair)


if __name__ == "__main__":
    main()
