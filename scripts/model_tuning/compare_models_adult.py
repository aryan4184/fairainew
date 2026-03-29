
import sys
import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset_defer.AdultDataset import Adult

def evaluate_model(name, clf, X, y, cv=3):
    print(f"\n--- Training {name} ---")
    start = time.time()
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    end = time.time()
    
    mean_acc = scores.mean()
    std_acc = scores.std()
    print(f"{name} CV Accuracy: {mean_acc:.4f} (+/- {std_acc*2:.4f})")
    print(f"Time taken: {end - start:.2f}s")
    return mean_acc, std_acc

def main():
    print("Loading AdultDataset...")
    device = torch.device("cpu")
    # Using default parameters
    dataset = Adult(data_dir="data", device=device)
    
  
    X_train, y_train = [], []
    for x, y, h, d in dataset.data_train_loader:
        X_train.append(x.numpy())
        y_train.append(y.numpy())
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    print(f"\nTraining Data Shape: {X_train.shape}")
    print(f"Class Distribution: {Counter(y_train)}")
    
    counts = Counter(y_train)
    majority_acc = max(counts.values()) / sum(counts.values())
    print(f"Majority Class Baseline: {majority_acc:.4f}")
    
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest (Default)", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Random Forest (Balanced)", RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("MLP (sklearn)", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    ]
    
    results = []
    
    for name, clf in models:
        mean, std = evaluate_model(name, clf, X_train, y_train)
        results.append({
            "Model": name,
            "Mean Accuracy": mean,
            "Std Dev": std
        })
        
    print("\n\n--- Summary Table ---")
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="Mean Accuracy", ascending=False)
    print(df_res.to_string(index=False))
    
    print(f"\nBaseline (Majority Class): {majority_acc:.4f}")

if __name__ == "__main__":
    main()
