
import pandas as pd
import os
import sys

def main():
    csv_path = "data/adult_reconstruction.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    
    # Simulate NewAdultDataset logic
    df_features = df.drop(['income'], axis=1)
    
    df_current = df.drop(['income', 'gender'], axis=1)
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    df_current = pd.get_dummies(df_current, columns=categorical_cols, drop_first=True)
    
    print("--- Current NewAdultDataset Features ---")
    print(f"Total: {len(df_current.columns)}")
    print("\n".join(df_current.columns.tolist()))

if __name__ == "__main__":
    main()
