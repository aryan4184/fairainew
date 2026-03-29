
import sys
import os
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.optim as optim

sys.path.append("..")
from dataset_defer.AdultDataset import Adult
from networks.linear_net import LinearNet
from methods.costcombination import PL_Combine_Cost
from methods.combination import PL_Combine
from methods.faircomb import PL_Combine_Fair
from helpers.validation_utils import compute_bound_components, bootstrap_metrics

# --- Configuration ---
DATA_DIR = "../data" # Adjust if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "../Results/validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_and_eval(method_name, dataset, device, epochs=10, human_cost=1.0, fairness_cost=20.0):
    """
    Trains a method and returns the test set predictions and deferrals.
    """
    model = LinearNet(dataset.d, 4).to(device)
    optimizer = optim.Adam
    lr = 1e-2
    
    if method_name == "Cost":
        method = PL_Combine_Cost(model, device, human_cost=human_cost)
    elif method_name == "All":
        method = PL_Combine(model, device)
    elif method_name == "Fair":
        method = PL_Combine_Fair(model, device, fairness_cost=fairness_cost, human_cost=human_cost)
    else:
        raise ValueError(f"Unknown method {method_name}")

    # Train
    method.fit(
        dataset.data_train_loader,
        dataset.data_val_loader,
        dataset.data_test_loader, 
        epochs=epochs,
        optimizer=optimizer,
        lr=lr,
        verbose=False
    )
    
    # Eval
    res = method.test(dataset.data_test_loader)
    return res

def run_test_group_1_and_2(dataset):
    """
    Test Group 1: Bound Check
    Test Group 2: Bootstrap Robustness
    """
    results = []
    
    modes = ["Cost", "Fair", "All"]
    
    print("\n--- Running Test Groups 1 & 2 (Bound Check & Bootstrap) ---")
    
    for mode in modes:
        print(f"Testing Mode: {mode}")
        start_res = train_and_eval(mode, dataset, DEVICE, epochs=20)
        
        # Prepare arrays
        y_true = start_res["labels"]
        # System predictions
        if "combined_preds" in start_res:
            y_pred = start_res["combined_preds"]
        else:
            # Fallback if method doesn't return combined_preds directly in dict (likely does)
            y_pred = start_res["preds"] # Should not happen for deferral methods
            
        demos = start_res["demographics"]
        
        # Standard Bound Check
        bound_metrics = compute_bound_components(y_true, y_pred, demos)
        bound_metrics["mode"] = mode
        bound_metrics["type"] = "System"
        results.append(bound_metrics)
        
        print(f"  LHS: {bound_metrics['lhs']:.4f}, RHS: {bound_metrics['rhs']:.4f}, Holds: {bound_metrics['holds']}")
        
        # Bootstrap
        boot_metrics = bootstrap_metrics(y_true, y_pred, demos, n_bootstrap=200)
        boot_metrics["mode"] = mode
        boot_metrics["type"] = "System (Bootstrap)"
        results.append(boot_metrics)
        
        print(f"  Bootstrap LHS: {boot_metrics['lhs_mean']:.4f} +/- {boot_metrics['lhs_std']:.4f}")
        print(f"  Violations: {boot_metrics['violation_rate'] * 100:.1f}%")

        # Also check Model-only and Human-only for baselines (Test Group 4)
        # Model
        y_pred_m = start_res["preds"]
        m_metrics = compute_bound_components(y_true, y_pred_m, demos)
        m_metrics["mode"] = mode
        m_metrics["type"] = "Model-only"
        results.append(m_metrics)
        
        # Human
        y_pred_h = start_res["hum_preds"]
        h_metrics = compute_bound_components(y_true, y_pred_h, demos)
        h_metrics["mode"] = mode
        h_metrics["type"] = "Human-only"
        results.append(h_metrics)
        
    return results

def run_test_group_3(dataset):
    """
    Test Group 3: Sensitivity Analysis (Threshold & Deferral)
    """
    print("\n--- Running Test Group 3 (Sensitivity) ---")
    
    sensitivity_results = []

    
    costs = [1.0, 10.0, 50.0, 100.0]
    
    for fc in tqdm(costs, desc="Deferral Sensitivity"):
        res = train_and_eval("Fair", dataset, DEVICE, epochs=10, fairness_cost=fc)
        
        # Calculate metrics
        y_true = res["labels"]
        y_pred = res["combined_preds"]
        demos = res["demographics"]
        defers = res["defers"]
        
        deferral_rate = np.mean(defers)
        bound = compute_bound_components(y_true, y_pred, demos)
        
        sensitivity_results.append({
            "param": "fairness_cost",
            "value": fc,
            "deferral_rate": deferral_rate,
            "lhs": bound["lhs"],
            "rhs": bound["rhs"],
            "tightness": bound["tightness"],
            "holds": bound["holds"]
        })
        
    return sensitivity_results

def main():
    # Load dataset once
    print("Loading dataset...")
    # Initialize dataset with fixed seed for reproducibility
    dataset = Adult(DATA_DIR, DEVICE)
    
    # Run tests
    results_g12 = run_test_group_1_and_2(dataset)
    results_g3 = run_test_group_3(dataset)
    
    # Save results
    all_results = {
        "group_1_and_2": results_g12,
        "group_3": results_g3
    }
    
    with open(os.path.join(OUTPUT_DIR, "validation_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Results saved to {OUTPUT_DIR}/validation_results.json")
    
    # Generate Plots
    df_g1 = pd.DataFrame([r for r in results_g12 if "mean" not in r.keys()]) # Filter out bootstrap aggregates
    
    plt.figure(figsize=(10, 6))
    melted = df_g1.melt(id_vars=["mode", "type"], value_vars=["lhs", "rhs"], var_name="Metric", value_name="Value")
    sns.barplot(data=melted, x="mode", y="Value", hue="Metric")
    plt.title("Bound Check (LHS vs RHS) across Modes")
    plt.savefig(os.path.join(OUTPUT_DIR, "bound_check.png"))
    
    # Tightness vs Deferral Rate (Group 3)
    df_g3 = pd.DataFrame(results_g3)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df_g3, x="deferral_rate", y="tightness")
    plt.plot(df_g3["deferral_rate"], df_g3["tightness"], '--')
    plt.xlabel("Deferral Rate")
    plt.ylabel("Tightness (LHS/RHS)")
    plt.title("Tightness Sensitivity to Deferral Rate")
    plt.savefig(os.path.join(OUTPUT_DIR, "sensitivity_deferral.png"))
    
    print("Plots generated.")

if __name__ == "__main__":
    main()
