import warnings
import numpy as np
import torch
from torch import nn, optim
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm.auto import tqdm


# Temperature Scaling Calibrator (cluster-wise)

class TSCalibrator:
    """
    Cluster-wise Temperature Scaling (Guo et al., 2017).

    Learns a separate temperature parameter per cluster of logits.
    """

    def __init__(self, temperature=1.0, n_clusters=14):
        self.temperature = []
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.loss_trace = None

    def fitHelper(self, logits, y):
        """
        Fit temperature scaling for a single cluster using NLL.
        """
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
                    warnings.warn("Temperature scaling may not have converged")
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
        try:
            clusters = self.kmeans.predict(logits)
        except NotFittedError:
            warnings.warn("KMeans calibrator not fitted; returning original probabilities")
            return probs

        for i in range(logits.shape[0]):
            probs[i] = probs[i] ** (1.0 / self.temperature[clusters[i]])

        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs
    
# Oracle Cost-Sensitive Combiner

class OracleCombiner:
    """
    Cost-sensitive P+L oracle combiner.

    Uses ground-truth labels to learn an optimal deferral policy.
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
        """
        Fit human confusion matrix and calibrator.
        """
        print(f"DEBUG: model_probs shape: {model_probs.shape}")
        self.n_cls = model_probs.shape[1]

        # Estimate P(h | Y)
        conf = confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls)).T
        conf = np.clip(conf, self.eps, None)
        conf /= np.sum(conf, axis=0, keepdims=True)
        self.confusion_matrix = conf

        self.fit_calibrator(model_probs, y_true)

    def fit_bayesian(self, model_probs, y_h, y_true, alpha=0.1, beta=0.1):
        """
        Bayesian smoothing of human confusion matrix.
        """
        self.n_cls = model_probs.shape[1]
        prior = np.eye(self.n_cls) * alpha + (1 - np.eye(self.n_cls)) * beta

        conf = confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        conf = (conf + prior).T
        conf /= np.sum(conf, axis=0, keepdims=True)
        self.confusion_matrix = conf

        self.fit_calibrator(model_probs, y_true)


    def fit_calibrator(self, model_probs, y_true):
        logits = np.log(np.clip(model_probs, self.eps, 1))
        self.calibrator.fit(logits, y_true)

    def combine_proba(self, model_probs, y_h, y_true, miss_cost=9, human_cost=1):
        """
        Cost-optimal combination of model and human predictions.
        """
        n, k = model_probs.shape
        if self.n_cls is None:
            self.n_cls = k

        calibrated = self.calibrate(model_probs)
        model_preds = np.argmax(calibrated, axis=1)

        y_comb = np.zeros((n, self.n_cls))
        defers = np.zeros(n)

        # If not fitted yet (training loop validation), default to model predictions
        if self.confusion_matrix is None:
            return calibrated, defers, {}

        human_correct = model_correct = 0
        human_used = model_used = 0

        for i in range(n):
            expected_model_cost = miss_cost * (1 - calibrated[i, model_preds[i]])

            if human_cost <= expected_model_cost:
                y_comb[i] = calibrated[i] * self.confusion_matrix[y_h[i]]
                defers[i] = 1
                human_used += 1
                human_correct += int(np.argmax(y_comb[i]) == y_true[i])
            else:
                y_comb[i] = calibrated[i]
                model_used += 1
                model_correct += int(np.argmax(y_comb[i]) == y_true[i])

            if np.allclose(y_comb[i], 0):
                y_comb[i] = np.ones(self.n_cls) / self.n_cls

        y_comb /= np.sum(y_comb, axis=1, keepdims=True)


        result = {
            "Combined accuracy": (human_correct + model_correct) / n,
            "Human accuracy": human_correct / human_used if human_used else 0,
            "Model accuracy": model_correct / model_used if model_used else 0,
            "Deferral rate": np.mean(defers),
            "Total cost": miss_cost * (model_used - model_correct) + human_cost * human_used,
        }

        return y_comb, defers, result
    
    def combine(self, model_probs, y_h, y_true, miss_cost=9, human_cost=1):
        if self.n_cls is None:
            self.n_cls = model_probs.shape[1]
            
        y_soft, defers, result = self.combine_proba(
            model_probs, y_h, y_true, miss_cost, human_cost
        )
        return np.argmax(y_soft, axis=1), result
    