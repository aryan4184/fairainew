
import sys
import os
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset_defer.hatespeech import HateSpeech
from networks.linear_net import LinearNet
from methods.faircomb import PL_Combine_Fair
from helpers.utils import accuracy

# Configure logging
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    print("Initializing HateSpeech dataset...")
    # Using same parameters as in ablation script
    dataset = HateSpeech(
        data_dir="data",
        embed_texts=True,
        include_demographics=False,
        expert_type="random_annotator",
        device=device
    )

    print("Dataset loaded.")
    
    model = LinearNet(dataset.d, 3).to(device)

    pl_fair = PL_Combine_Fair(model, device, k=10) 
    

    print("Fitting combiner (skipping full model training for speed)...")
    

    pl_fair.fit_combiner(dataset.data_train_loader)
    

    

    print("\n--- Test Case 1: Experiment Defaults (r=0.99, human_cost=1.0) ---")
    pl_fair.human_cost = 1.0
    output = pl_fair.test(dataset.data_test_loader, fairness_cost=0.99)
    print(f"Deferral Rate: {np.mean(output['defers']):.4f}")
    print(f"System Accuracy: {np.mean(output['combined_preds'] == output['labels']):.4f}")


    print("\n--- Test Case 2: Forced Deferral (r=0.99, human_cost=0.0) ---")
    pl_fair.human_cost = 0.0
    output = pl_fair.test(dataset.data_test_loader, fairness_cost=0.99)
    print(f"Deferral Rate: {np.mean(output['defers']):.4f}")
    print(f"System Accuracy: {np.mean(output['combined_preds'] == output['labels']):.4f}")
    print(f"Human Accuracy: {np.mean(output['hum_preds'] == output['labels']):.4f}")

if __name__ == "__main__":
    main()
