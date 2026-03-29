
import pandas as pd
import os

def main():
    csv_path = "data/adult_reconstruction.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    income = df['income']
    
 
    print("Searching for thresholds matching user proportions:")
    print("Target High (Class 0): ~5.1%")
    print("Target Mid  (Class 2): ~18.9%")
    print("Target Low  (Class 1): ~75.9%")
    

    high_thresh = income.quantile(1 - 0.051)

    mid_thresh = income.quantile(1 - 0.24)
    
    print(f"\nEstimated Thresholds:")
    print(f"High Threshold (>X): {high_thresh}")
    print(f"Mid Threshold (>Y): {mid_thresh}")
    
    
    high_mask = income > high_thresh
    mid_mask = (income > mid_thresh) & (income <= high_thresh)
    low_mask = income <= mid_thresh
    
    print("\nResulting Counts on current file:")
    print(f"High: {high_mask.sum()} ({high_mask.mean():.2%})")
    print(f"Mid:  {mid_mask.sum()} ({mid_mask.mean():.2%})")
    print(f"Low:  {low_mask.sum()} ({low_mask.mean():.2%})")

if __name__ == "__main__":
    main()
