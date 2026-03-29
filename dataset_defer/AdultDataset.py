import torch
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import random_split, TensorDataset, DataLoader
import sys

sys.path.append("../")

from .basedataset import *
from .human import *


class Adult(BaseDataset):
    def __init__(self, data_dir, device, test_split=0.2, val_split=0.1, batch_size=1000, random_seed=42):
        super().__init__()

        self.data_dir = data_dir
        self.device = device
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.n_dataset = 3  # Number of income classes
        self.generate_data()

    def generate_data(self):
        df = pd.read_csv(os.path.join(self.data_dir, "adult_reconstruction.csv"))
        t1, t2 = 25000, 60000
        df['income'] = pd.cut(df['income'], 
                            bins=[0, t1, t2, float('inf')], 
                            labels=[0, 1, 2])
        
        categorical_cols = ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race', 'native-country', 'gender']
        self.encoders = {col: LabelEncoder() for col in categorical_cols}

        for col in categorical_cols:
            df[col] = self.encoders[col].fit_transform(df[col])

        demographics = df['gender'].values
        y = df['income'].values
        X = df.drop(['income', 'gender'], axis=1).values

        human_labels = self._create_human_labels(y, accuracy=0.65)

        total_size = len(X)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - test_size - val_size

        from torch.utils.data import Subset
        # Get random split indices FIRST
        dummy_dataset = TensorDataset(torch.zeros(total_size))
        train_sub, val_sub, test_sub = random_split(
            dummy_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )

        self.scaler = StandardScaler()
        self.scaler.fit(X[train_sub.indices])
        X = self.scaler.transform(X)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y.astype('int64'))
        human_tensor = torch.LongTensor(human_labels.astype('int64'))
        demo_tensor = torch.LongTensor(demographics.astype('int64'))

        full_dataset = TensorDataset(X_tensor, y_tensor, human_tensor, demo_tensor)
        self.train_dataset = Subset(full_dataset, train_sub.indices)
        self.val_dataset = Subset(full_dataset, val_sub.indices)
        self.test_dataset = Subset(full_dataset, test_sub.indices)

        self.d = X.shape[1]  # Feature dimension
        self.n_dataset = 3   # Number of classes

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










