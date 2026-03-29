
import pandas as pd
import os

def main():
    csv_path = "data/adult_reconstruction.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    print("\nIncome Stats:")
    print(df['income'].describe())
    
    t1, t2 = 25000, 60000
    print(f"\nApplying bins: [0, {t1}, {t2}, inf]")
    
    df['income_bin'] = pd.cut(df['income'], bins=[0, t1, t2, float('inf')], labels=[0, 1, 2])
    
    counts = df['income_bin'].value_counts().sort_index()
    print("\nClass Counts (AdultDataset logic):")
    print(counts)
    
    print("\nClass Proportions:")
    print(counts / len(df))

if __name__ == "__main__":
    main()
