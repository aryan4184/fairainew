import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

def test_balanced():
    
    df = pd.read_csv("data/adult_reconstruction.csv")
    
    def income_class(x):
        if x < 25000: return "Low"
        elif x <= 60000: return "Mid"
        else: return "High"
    
    df["income_class"] = df["income"].apply(income_class)
    le = LabelEncoder()
    df["income_class_encoded"] = le.fit_transform(df["income_class"])
    
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    

    X = df.drop(columns=["income", "income_class", "income_class_encoded"])
    y = df["income_class_encoded"]
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n--- Original Training Distribution ---")
    print(y_train.value_counts().sort_index())


    train_data = pd.concat([X_train, y_train], axis=1)
    
    class_0 = train_data[train_data.income_class_encoded == 0] # High
    class_1 = train_data[train_data.income_class_encoded == 1] # Low
    class_2 = train_data[train_data.income_class_encoded == 2] # Mid
    
    min_size = min(len(class_0), len(class_1), len(class_2))
    print(f"\nResampling all classes to size: {min_size}")
    
    class_0_down = resample(class_0, replace=False, n_samples=min_size, random_state=42)
    class_1_down = resample(class_1, replace=False, n_samples=min_size, random_state=42)
    class_2_down = resample(class_2, replace=False, n_samples=min_size, random_state=42)
    
    train_balanced = pd.concat([class_0_down, class_1_down, class_2_down])
    
    X_train_bal = train_balanced.drop(columns=["income_class_encoded"])
    y_train_bal = train_balanced["income_class_encoded"]
    
    print("\n--- Balanced Training Distribution ---")
    print(y_train_bal.value_counts().sort_index())
    
    print("\nTraining HistGradientBoostingClassifier on Balanced Data...")
    model = HistGradientBoostingClassifier(random_state=42, max_iter=200)
    model.fit(X_train_bal, y_train_bal)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy (on imbalanced test set): {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    test_balanced()
