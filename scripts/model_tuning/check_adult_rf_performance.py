
import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset_defer.AdultDataset import Adult

def main():
    print("Loading AdultDataset...")
    device = torch.device("cpu")

    dataset = Adult(data_dir="data", device=device)
    
    X_train, y_train = [], []
    for x, y, h, d in dataset.data_train_loader:
        X_train.append(x.numpy())
        y_train.append(y.numpy())
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    # Class Distribution
    print(f"\nTraining Data Shape: {X_train.shape}")
    print(f"Class Distribution: {Counter(y_train)}")
    
    # baseline accuracy (majority class)
    counts = Counter(y_train)
    majority_acc = max(counts.values()) / sum(counts.values())
    print(f"Majority Class Baseline: {majority_acc:.4f}")
    
    print("\n--- Training Default Random Forest ---")
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_default = cross_val_score(rf_default, X_train, y_train, cv=3)
    print(f"Default RF CV Accuracy: {scores_default.mean():.4f} (+/- {scores_default.std()*2:.4f})")
    
    print("\n--- Training Balanced Random Forest ---")
    rf_balanced = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    scores_balanced = cross_val_score(rf_balanced, X_train, y_train, cv=3)
    print(f"Balanced RF CV Accuracy: {scores_balanced.mean():.4f} (+/- {scores_balanced.std()*2:.4f})")
    
    print("\n--- Training Tuned Random Forest (min_samples_leaf=5) ---")
    rf_tuned = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
    scores_tuned = cross_val_score(rf_tuned, X_train, y_train, cv=3)
    print(f"Tuned RF CV Accuracy: {scores_tuned.mean():.4f} (+/- {scores_tuned.std()*2:.4f})")
    
if __name__ == "__main__":
    main()
