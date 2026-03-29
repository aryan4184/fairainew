import numpy as np 
import pandas as pd 

def compute_label_counts(human_predictions, label_mapping):
    """
    Compute label counts for each instance based on multiple human predictions.
    
    Args:
        human_predictions (pd.DataFrame): N x K matrix where each row represents 
                                          multiple human annotations.
        label_mapping (dict): Dictionary mapping possible label values 
                              (e.g., {0: "negative", 1: "positive"}).
    
    Returns:
        pd.DataFrame: DataFrame with added "label_counts" column.
    """
    label_counts = human_predictions.apply(lambda row: row.value_counts().reindex(label_mapping.keys(), fill_value=0).to_list(), axis=1)
    return label_counts

def compute_label_distribution(label_counts):
    """
    Convert label counts into a probability distribution.
    
    Args:
        label_counts (list of lists): List where each entry is a count vector 
                                      for a sample (e.g., [3, 5, 2] for 3x label 0, 5x label 1, 2x label 2).
    
    Returns:
        list of np.array: Probability distributions per instance.
    """
    return [np.array(counts) / sum(counts) for counts in label_counts]
