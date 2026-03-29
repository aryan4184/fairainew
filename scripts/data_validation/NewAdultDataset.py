
import torch
import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import random_split, TensorDataset, DataLoader
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset_defer.basedataset import BaseDataset

class NewAdultDataset(BaseDataset):
    """
    NewAdultDataset with One-Hot Encoding for categorical features.
    Maintains the same logic as AdultDataset but improves feature representation.
    """
    def __init__(self, data_dir, device, test_split=0.2, val_split=0.1, batch_size=1000):
        super().__init__()

        self.data_dir = data_dir
        self.device = device
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 3  # Number of income classes
        self.generate_data()

    def generate_data(self):
        csv_path = os.path.join(self.data_dir, "adult_reconstruction.csv")
        if not os.path.exists(csv_path):
            # Fallback if running from nt/
            csv_path = os.path.join(self.data_dir, "../data/adult_reconstruction.csv")
            
        df = pd.read_csv(csv_path)
        # Thresholds derived to match user distribution (High~5%, Mid~19%, Low~76%)
        t1, t2 = 49800, 98002
        
        # Initial Cut: 0=Low, 1=Mid, 2=High
        df['income'] = pd.cut(df['income'], 
                            bins=[0, t1, t2, float('inf')], 
                            labels=[0, 1, 2])
        

        label_map = {0: 1, 1: 2, 2: 0}
        df['income'] = df['income'].map(label_map)
        
        # Define categorical columns
        categorical_cols = ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race', 'native-country', 'gender']
        
        # Extract Target (y)
        y = df['income'].values
        
        le_gender = LabelEncoder()
        demographics = le_gender.fit_transform(df['gender'])
        
        
        df_features = df.drop(['income'], axis=1)
        
    
        feature_categorical_cols = categorical_cols
        
        df_features = pd.get_dummies(df_features, columns=feature_categorical_cols, drop_first=True)
        
        X = df_features.values
        
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        human_labels = self._create_human_labels(y, accuracy=0.65)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y.astype('int64'))
        human_tensor = torch.LongTensor(human_labels.astype('int64'))
        demo_tensor = torch.LongTensor(demographics.astype('int64'))

        full_dataset = TensorDataset(X_tensor, y_tensor, human_tensor, demo_tensor)

        total_size = len(full_dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - test_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.d = X.shape[1]  
        self.n_dataset = 3 

        pin_memory = (self.device.type == 'cuda')

        self.data_train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=pin_memory
        )

        self.data_val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=pin_memory
        )

        self.data_test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=pin_memory
        )


    def _create_human_labels(self, y, accuracy=0.8):
        """Create synthetic human labels with specified accuracy"""
        human_labels = np.copy(y)
        mask = np.random.rand(len(y)) > accuracy

        for i in np.where(mask)[0]:
            possible_labels = [l for l in np.unique(y) if l != y[i]]
            human_labels[i] = np.random.choice(possible_labels)

        return human_labels
