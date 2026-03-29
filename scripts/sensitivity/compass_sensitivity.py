
import logging
import sys
import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
import fairlearn.metrics as metrics
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset_defer.broward import BrowardDataset
from networks.linear_net import LinearNet
from methods.faircomb import PL_Combine_Fair
from helpers.validation_utils import compute_bound_components
from helpers.metrics import accuracy_gap_per_group

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "compass_sensitivity.log")

file_handler = logging.FileHandler(OUTPUT_FILE, mode='w')
stdout_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO, 
    format='%(message)s',
    handlers=[file_handler, stdout_handler]
)

def combine_defer(preds, h_preds, defers):
    """Combine model and human predictions using deferral decisions"""
    return preds * (1 - defers) + h_preds * defers

def print_metrics(data, class_num=2, combine_method="defer"):
    """
    Compute and print fairness + accuracy metrics
    for each class and demographic group
    """
    res = {}
    
    if torch.is_tensor(data["preds"]): data["preds"] = data["preds"].cpu().numpy()
    if torch.is_tensor(data["labels"]): data["labels"] = data["labels"].cpu().numpy()
    if torch.is_tensor(data["hum_preds"]): data["hum_preds"] = data["hum_preds"].cpu().numpy()
    if torch.is_tensor(data["defers"]): data["defers"] = data["defers"].cpu().numpy()
    if torch.is_tensor(data["demographics"]): data["demographics"] = data["demographics"].cpu().numpy()
    if "combined_preds" in data and torch.is_tensor(data["combined_preds"]):
         data["combined_preds"] = data["combined_preds"].cpu().numpy()


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
            if mask.sum() == 0: continue

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

    res["deferral rate"] = data["defers"].mean()
    res["model accuracy"] = (data["preds"] == data["labels"]).mean()
    res["human accuracy"] = (data["hum_preds"] == data["labels"]).mean()

    if combine_method == "defer":
        combined_preds = combine_defer(data["preds"], data["hum_preds"], data["defers"])
    else:
        combined_preds = np.array(data["combined_preds"])

    res["combined accuracy"] = (combined_preds == data["labels"]).mean()
   
    acc_gap_res = accuracy_gap_per_group(
        data,
        use_combined=(combine_method == "defer")
    )

    res.update(acc_gap_res)

    for k, v in res.items():
        logging.info(f"{k} : {v}")

    return res

def train_and_eval_fair(dataset, device, fairness_cost=10.0, human_cost=1.0, test_loader_override=None):
    """
    Trains PL_Combine_Fair and evaluates.
    """
    model = LinearNet(dataset.d, 4).to(device)
    optimizer = optim.Adam
    lr = 1e-2
    
    method = PL_Combine_Fair(
        model, 
        device, 
        fairness_cost=fairness_cost, 
        human_cost=human_cost,
        plotting_interval=10 
    )

    # Train
    method.fit(
        dataset.data_train_loader,
        dataset.data_val_loader,
        dataset.data_test_loader, 
        epochs=100,
        optimizer=optimizer,
        lr=lr,
        verbose=False # logging handles it or silent
    )
    
    loader = test_loader_override if test_loader_override else dataset.data_test_loader
    res = method.test(loader)
    return res

def compute_metrics_tuple(y_true, y_pred, demos):
    """Returns (Accuracy, EOD)"""
    bound = compute_bound_components(y_true, y_pred, demos)
    acc = np.mean(y_true == y_pred)
    # EOD = max(|dTPR|, |dFPR|)
    eod = max(bound["delta_tpr"], bound["delta_fpr"])
    return acc, eod

def get_metrics_line(label, res, extra_info=""):
    y_true = res["labels"]
    y_pred = res["combined_preds"]
    demos = res["demographics"]
    defers = res["defers"]
    
    bound = compute_bound_components(y_true, y_pred, demos)
    deferral_rate = np.mean(defers)
    
    lhs = f"{bound['lhs']:.4f}"
    rhs = f"{bound['rhs']:.4f}"
    tight = f"{bound['tightness']:.4f}"
    viol = f"{bound['holds']}"
    def_rate = f"{deferral_rate:.4f}"
    
    return f"{label:<25} | {lhs:<8} | {rhs:<8} | {tight:<10} | {viol:<5} | {def_rate:<8} | {extra_info}"

def experiment_a(dataset):
    lines = []
    lines.append("\n" + "="*80)
    lines.append("(A) DEFERRAL RATE SENSITIVITY")
    lines.append("="*80)
    lines.append(f"{'Condition':<25} | {'LHS':<8} | {'RHS':<8} | {'Tightness':<10} | {'Holds':<5} | {'Defer%':<8} |")
    lines.append("-" * 80)
    
    configs = [
        ("Low Deferral", 3.0),
        ("Med Deferral", 8.0),
        ("High Deferral", 20.0)
    ]
    
    detailed_lines = []
    
    for name, fc in configs:
        logging.info(f"Running Exp A: {name} (fc={fc})...")
        res = train_and_eval_fair(dataset, DEVICE, fairness_cost=fc)
        line = get_metrics_line(name, res, f"Target fc={fc}")
        lines.append(line)
        
        logging.info(f"\n--- Metrics for {name} ---")
        print_metrics(res, class_num=2, combine_method="PL")
        
        y_true = res["labels"]
        demos = res["demographics"]
        
        m_acc, m_eod = compute_metrics_tuple(y_true, res["preds"], demos)
        h_acc, h_eod = compute_metrics_tuple(y_true, res["hum_preds"], demos)
        s_acc, s_eod = compute_metrics_tuple(y_true, res["combined_preds"], demos)
        
        detailed_lines.append(f"{name:<25} | {m_acc:.3f} / {m_eod:.3f} | {h_acc:.3f} / {h_eod:.3f} | {s_acc:.3f} / {s_eod:.3f}")

    lines.append("")
    lines.append("="*80)
    lines.append("DETAILED METRICS (ACC / EOD)")
    lines.append("-" * 80)
    lines.append(f"{'Condition':<25} | {'Model':<15} | {'Human':<15} | {'System':<15}")
    lines.append(f"{'':<25} | {'Acc / EOD':<15} | {'Acc / EOD':<15} | {'Acc / EOD':<15}")
    lines.append("-" * 80)
    lines.extend(detailed_lines)

    return lines

def experiment_b(dataset):
    lines = []
    lines.append("\n" + "="*80)
    lines.append("(B) HUMAN COST VS DEFERRAL COST TRADEOFF")
    lines.append("="*80)
    lines.append(f"{'Condition':<25} | {'LHS':<8} | {'RHS':<8} | {'Tightness':<10} | {'Holds':<5} | {'Defer%':<8} | {'Params (HC/FC)'}")
    lines.append("-" * 80)
    
    configs = [
        ("Push Defer", 0.1, 15.0),
        ("Balanced", 1.0, 8.0),
        ("Push Model", 10.0, 4.0)
    ]
    
    for name, hc, fc in configs:
        logging.info(f"Running Exp B: {name}...")
        res = train_and_eval_fair(dataset, DEVICE, human_cost=hc, fairness_cost=fc)
        line = get_metrics_line(name, res, f"hc={hc}, fc={fc}")
        lines.append(line)
        
        logging.info(f"\n--- Metrics for {name} ---")
        print_metrics(res, class_num=2, combine_method="PL")
        
    return lines

def experiment_c(dataset):
    lines = []
    lines.append("\n" + "="*80)
    lines.append("(C) k-NN HUMAN ORACLE SENSITIVITY")
    lines.append("="*80)
    lines.append(f"{'k Value':<25} | {'LHS':<8} | {'RHS':<8} | {'Tightness':<10} | {'Holds':<5} | {'Defer%':<8} | {'Human Acc'}")
    lines.append("-" * 80)
    
    # Get Training Data for fitting the Oracle
    train_loader = dataset.data_train_loader
    X_train, y_train = [], []
    for x, y, h, d in train_loader:
        X_train.append(x.numpy())
        y_train.append(y.numpy())
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    #Get Test Data Base
    test_loader = dataset.data_test_loader
    X_test_list, y_test_list, d_test_list = [], [], []
    for x, y, h, d in test_loader:
        X_test_list.append(x)
        y_test_list.append(y)
        d_test_list.append(d)
        
    X_test_t = torch.cat(X_test_list)
    y_test_t = torch.cat(y_test_list)
    d_test_t = torch.cat(d_test_list)
    X_test_np = X_test_t.numpy()
    
    k_values = [1, 15]
    
    for k in k_values:
        logging.info(f"Running Exp C: k={k}...")
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        h_pred_np = knn.predict(X_test_np)
        h_pred_t = torch.LongTensor(h_pred_np)
        
        h_acc = np.mean(h_pred_np == y_test_t.numpy())
        
        new_dataset = TensorDataset(X_test_t, y_test_t, h_pred_t, d_test_t)
        new_loader = DataLoader(new_dataset, batch_size=dataset.batch_size, shuffle=False)
        
        res = train_and_eval_fair(dataset, DEVICE, test_loader_override=new_loader)
        
        line = get_metrics_line(f"k={k}", res, f"H_Acc={h_acc:.3f}")
        lines.append(line)
        
        logging.info(f"\n--- Metrics for k={k} ---")
        print_metrics(res, class_num=2, combine_method="PL")
        
    return lines

def main():
    logging.info("Loading Broward/COMPAS Dataset...")
    basedir = os.path.dirname(__file__)
    data_dir = os.path.join(basedir, "../data")
    
    dataset = BrowardDataset(data_dir, test_split=0.2, val_split=0.1)
    
    all_lines = []
    
    all_lines.extend(experiment_a(dataset))
    all_lines.extend(experiment_b(dataset))
    all_lines.extend(experiment_c(dataset))
    
    report = "\n".join(all_lines)
    logging.info(report)
    
    logging.info(f"\nReport saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
