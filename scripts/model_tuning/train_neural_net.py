import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import copy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 5


def load_data():
    df = pd.read_csv("data/adult_reconstruction.csv")

    def income_class(x):
        if x < 25000: return "Low"
        elif x <= 60000: return "Mid"
        else: return "High"

    df["income_class"] = df["income"].apply(income_class)
    le_target = LabelEncoder()
    df["income_class_encoded"] = le_target.fit_transform(df["income_class"])
    
    print("Classes:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

    target_col = "income_class_encoded"
    drop_cols = ["income", "income_class", "income_class_encoded"]
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    cat_cols = [c for c in cat_cols if c not in drop_cols]
    
    num_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    num_cols = [c for c in num_cols if c not in drop_cols]
    
    print(f"Categorical features: {len(cat_cols)}")
    print(f"Numerical features: {len(num_cols)}")

    cat_dims = []
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        cat_dims.append(len(le.classes_))

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test, cat_cols, num_cols, cat_dims, le_target


class AdultDataset(Dataset):
    def __init__(self, X, y, cat_cols, num_cols):
        self.X_cat = torch.tensor(X[cat_cols].values, dtype=torch.long)
        self.X_num = torch.tensor(X[num_cols].values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


# Neural Network Model
class EntityEmbeddingNet(nn.Module):
    def __init__(self, cat_dims, num_dim, output_dim, emb_dims=None):
        super().__init__()
        
        # Embeddings
        self.embeddings = nn.ModuleList()
        if emb_dims is None:
            # Rule of thumb: min(50, (dim+1)//2)
            emb_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]
            
        for dim, emb_dim in zip(cat_dims, emb_dims):
            self.embeddings.append(nn.Embedding(dim, emb_dim))
            
        total_emb_dim = sum(emb_dims)
        
        # Main MLP
        input_dim = total_emb_dim + num_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, output_dim)
        )

    def forward(self, x_cat, x_num):
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            embedded.append(emb_layer(x_cat[:, i]))
        
        x_emb = torch.cat(embedded, dim=1)
        x = torch.cat([x_emb, x_num], dim=1)
        
        return self.mlp(x)


# Training Loop
def train_model():
    X_train, X_test, y_train, y_test, cat_cols, num_cols, cat_dims, le_target = load_data()
    
    class_counts = y_train.value_counts().sort_index().values
    total_samples = len(y_train)
    class_weights = torch.tensor(total_samples / (len(class_counts) * class_counts), dtype=torch.float32).to(DEVICE)
    print(f"Class Weights: {class_weights}")

    train_ds = AdultDataset(X_train, y_train, cat_cols, num_cols)
    test_ds = AdultDataset(X_test, y_test, cat_cols, num_cols)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = EntityEmbeddingNet(cat_dims, len(num_cols), 3).to(DEVICE)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss() # Optimized for raw accuracy
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for x_cat, x_num, labels in train_loader:
            x_cat, x_num, labels = x_cat.to(DEVICE), x_num.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x_cat, x_num)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x_cat.size(0)
            
        epoch_loss = running_loss / len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x_cat, x_num, labels in test_loader:
                x_cat, x_num, labels = x_cat.to(DEVICE), x_num.to(DEVICE), labels.to(DEVICE)
                outputs = model(x_cat, x_num)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * x_cat.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(test_ds)
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break
                
    # Final Evaluation
    model.load_state_dict(best_model_wts)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_cat, x_num, labels in test_loader:
            x_cat, x_num, labels = x_cat.to(DEVICE), x_num.to(DEVICE), labels.to(DEVICE)
            outputs = model(x_cat, x_num)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\nFinal Test Metrics:")
    print(classification_report(all_labels, all_preds, digits=4))
    print(f"Final Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

if __name__ == "__main__":
    train_model()
