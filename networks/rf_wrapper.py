
import torch
import torch.nn as nn
import numpy as np

class RFWrapper(nn.Module):
    """
    Wraps an sklearn RandomForestClassifier to behave like a PyTorch module.
    Returns logits (log_softmax) by taking log(predict_proba).
    """
    def __init__(self, clf, device):
        super().__init__()
        self.clf = clf
        self.device = device
        # Dummy parameter to satisfy optimizer initialization when epochs=0
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        # predict_proba returns [n_samples, n_classes]
        probs = self.clf.predict_proba(x_np)
        
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        
        # Convert to logits: log(p). 
        # When F.softmax is applied later, it will return approx p again.
        logits = np.log(probs)
        
        return torch.tensor(logits, dtype=torch.float32).to(self.device)
