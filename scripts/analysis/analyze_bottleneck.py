import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier

def analyze_bottleneck():
    # Load dataset
    df = pd.read_csv("data/adult_reconstruction.csv")
    
    # Create income classes
    def income_class(x):
        if x < 25000: return "Low"
        elif x <= 60000: return "Mid"
        else: return "High"
    
    df["income_class"] = df["income"].apply(income_class)
    le = LabelEncoder()
    df["income_class_encoded"] = le.fit_transform(df["income_class"])
    
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    
    # Prepare features
    X = df.drop(columns=["income", "income_class", "income_class_encoded"])
    y = df["income_class_encoded"]
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train high-performance model
    print("\nTraining HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(random_state=42, max_iter=200)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)
    
    print(f"\nTrain Accuracy: {acc_train:.4f}")
    print(f"Test Accuracy:  {acc_test:.4f}")
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Feature Importance (Permutation)
    print("\nCalculating Permutation Importance...")
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    
    print("\nTop 10 Important Features:")
    for i in sorted_idx[-10:]:
        print(f"{feature_names[i]}: {result.importances_mean[i]:.4f}")
        
    # Bayes Error Estimation (k-NN)
    print("\nEstimating Bayes Error using k-NN (k=5)...")
    # Scale for k-NN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    knn_acc = knn.score(X_test_scaled, y_test)
    print(f"k-NN (k=5) Accuracy: {knn_acc:.4f} (Proxy for local separability)")
    
    # PCA Visualization
    print("\nGenerating PCA plot...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(scatter, label='Income Class')
    plt.title('PCA Projection of Test Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('pca_plot.png')
    print("PCA plot saved to 'pca_plot.png'")

if __name__ == "__main__":
    analyze_bottleneck()
