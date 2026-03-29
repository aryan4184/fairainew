import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def test_smote():
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
    
    # Initial Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n--- Original Training Distribution ---")
    print(y_train.value_counts().sort_index())
    
    #SMOTE
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print("\n--- Balanced Training Distribution (SMOTE) ---")
    print(y_train_bal.value_counts().sort_index())
    
    # Train Model
    print("\nTraining HistGradientBoostingClassifier on SMOTE Data...")
    model = HistGradientBoostingClassifier(random_state=42, max_iter=200)
    model.fit(X_train_bal, y_train_bal)
    
    # Predict on UNTOUCHED Test Set
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy (on imbalanced test set): {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    test_smote()
