from tqdm.auto import tqdm
import warnings
import numpy as np

import torch
from torch import nn, optim
from torch.nn.functional import softmax

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Temperature Scaling Calibrator
class TSCalibrator:
    """
    Cluster-wise Temperature Scaling (Guo et al., 2017 inspired)

    Learns a separate temperature for each cluster of logits.
    """

    def __init__(self, temperature=1.0, n_clusters=14):
        self.temperature = []
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.loss_trace = None
        self.is_fitted = False

    def fitHelper(self, logits, y):
        """Fit temperature scaling for a single cluster."""
        n_classes = logits.shape[1]
        logits = torch.from_numpy(logits)
        y = torch.from_numpy(y)

        temperature = torch.tensor(1.0, requires_grad=True)

        nll = nn.CrossEntropyLoss()
        optimizer = optim.Adam([temperature], lr=0.05)

        num_steps = 7500
        grad_tol = 1e-3
        min_temp, max_temp = 1e-2, 1e4

        loss_trace = []
        step = 0

        while True:
            optimizer.zero_grad()
            loss = nll(logits / temperature, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                temperature.clamp_(min=min_temp, max=max_temp)

            loss_trace.append(loss.item())
            step += 1

            if step > num_steps or abs(temperature.grad) < grad_tol:
                if step > num_steps:
                    warnings.warn("TS did not fully converge")
                break

        self.loss_trace = loss_trace
        return temperature.item()

    def fit(self, model_logits, y_true):
        self.kmeans.fit(model_logits)

        logits_by_cluster = [[] for _ in range(self.n_clusters)]
        labels_by_cluster = [[] for _ in range(self.n_clusters)]
        self.temperature = [1.0] * self.n_clusters

        for i, c in enumerate(self.kmeans.labels_):
            logits_by_cluster[c].append(model_logits[i])
            labels_by_cluster[c].append(y_true[i])

        for c in range(self.n_clusters):
            self.temperature[c] = self.fitHelper(
                np.array(logits_by_cluster[c]),
                np.array(labels_by_cluster[c])
            )


    def calibrate(self, probs):
        probs = np.clip(probs, 1e-12, 1)
        logits = np.log(probs)
        if not hasattr(self.kmeans, "cluster_centers_"):
            raise RuntimeError(
                "TSCalibrator.calibrate() called before fit(). "
                "You must call AllCombiner.fit() first."
            )
        clusters = self.kmeans.predict(logits)

        for i in range(logits.shape[0]):
            probs[i] = probs[i] ** (1.0 / self.temperature[clusters[i]])

        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs
    
# P + L Combiner
class AllCombiner:
    """
    Implements the P+L (Prediction + Labeler) combination method.
    """

    def __init__(self, calibration_method="temperature scaling"):
        self.calibration_method = calibration_method
        self.calibrator = TSCalibrator()
        self.confusion_matrix = None
        self.n_cls = None
        self.eps = 1e-12
        self.use_cv = False

    def calibrate(self, model_probs):
        return self.calibrator.calibrate(model_probs)
    
    def fit(self, model_probs, y_h, y_true):
        self.n_cls = model_probs.shape[1]

        # Estimate human confusion matrix: P(h | Y)
        conf = confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls)).T
        conf = np.clip(conf, self.eps, None)
        conf /= np.sum(conf, axis=0, keepdims=True)
        self.confusion_matrix = conf

        self.fit_calibrator(model_probs, y_true)
        self.is_fitted = True

    def fit_calibrator(self, model_probs, y_true):
        model_probs = np.clip(model_probs, self.eps, 1)
        logits = np.log(model_probs)
        self.calibrator.fit(logits, y_true)

    def combine_proba(self, model_probs, y_h, y_true_te):
        """
        Compute P(Y | model, human) using calibrated probabilities and
        human confusion matrix.
        """
        n = model_probs.shape[0]
        calibrated = self.calibrate(model_probs)

        y_comb = np.zeros((n, self.n_cls))
        for i in range(n):
            y_comb[i] = calibrated[i] * self.confusion_matrix[y_h[i]]

            if np.allclose(y_comb[i], 0):
                y_comb[i] = np.ones(self.n_cls) / self.n_cls

        y_comb /= np.sum(y_comb, axis=1, keepdims=True)
        return y_comb, {}
    
    def combine(self, model_probs, y_h, y_true_te):
        y_soft, info = self.combine_proba(model_probs, y_h, y_true_te)
        return np.argmax(y_soft, axis=1), info