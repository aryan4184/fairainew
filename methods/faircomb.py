import time
import sys
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

sys.path.append("..")

from helpers.utils import AverageMeter, accuracy
from helpers.metrics import compute_classification_metrics
from baselines.basemethod import BaseMethod
from methods.allcombiner import AllCombiner

eps_cst = 1e-8


class PL_Combine_Fair(BaseMethod):
    """
    Fairness-aware learning-to-defer method.

    Defers to humans when local fairness risk exceeds the cost of deferral.
    Fairness is estimated using local k-NN disparities across demographic groups.
    """

    def __init__(
        self,
        model_class,
        device,
        plotting_interval=100,
        k=None,
        fairness_cost=9.0,
        human_cost=1.0,
    ):
        
        super().__init__()
        self.model_class = model_class
        self.device = device
        self.plotting_interval = plotting_interval

        self.combiner = AllCombiner()
        self.k = k
        self.fairness_cost = fairness_cost
        self.human_cost = human_cost

        self.nn_dem0 = None
        self.nn_dem1 = None

    # Train classifier
    def fit_epoch_class(self, dataloader, optimizer, verbose=True, epoch=1):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        self.model_class.train()
        for batch, (x, y, h, d) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            outputs = self.model_class(x)
            loss = F.cross_entropy(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(outputs.data, y, topk=(1,))[0]
            losses.update(loss.item(), x.size(0))
            top1.update(prec1.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if torch.isnan(loss):
                logging.warning("NaN loss encountered")
                break

            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    f"Epoch [{epoch}][{batch}/{len(dataloader)}] "
                    f"Loss {losses.avg:.4f} | Acc {top1.avg:.2f}"
                )

    # Fit combiner + neighborhood models
    def fit_combiner(self, dataloader):
        class_probs, y_true, y_h, y_pred = [], [], [], []
        feats_dem0, labels_dem0, preds_dem0 = [], [], []
        feats_dem1, labels_dem1, preds_dem1 = [], [], []

        if self.k is None:
            n = len(dataloader.dataset)
            self.k = max(5, min(30, int(math.log2(n))))
        logging.info(f"Using k = {self.k} for k-NN")

        self.model_class.eval()
        with torch.no_grad():
            for x, y, h, d in dataloader:
                x, y, h, d = x.to(self.device), y.to(self.device), h.to(self.device), d.to(self.device)

                probs = F.softmax(self.model_class(x), dim=1)
                _, preds = torch.max(probs, 1)

                class_probs.extend(probs.cpu().numpy())
                y_true.extend(y.cpu().numpy())
                y_h.extend(h.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                for i in range(x.size(0)):
                    if d[i].item() == 0:
                        feats_dem0.append(x[i].cpu().numpy())
                        labels_dem0.append(y[i].cpu().numpy())
                        preds_dem0.append(preds[i].cpu().numpy())
                    else:
                        feats_dem1.append(x[i].cpu().numpy())
                        labels_dem1.append(y[i].cpu().numpy())
                        preds_dem1.append(preds[i].cpu().numpy())

        self.combiner.fit(
            np.array(class_probs),
            np.array(y_h),
            np.array(y_true),
        )

        self.nn_dem0 = NearestNeighbors(n_neighbors=self.k).fit(feats_dem0)
        self.nn_dem1 = NearestNeighbors(n_neighbors=self.k).fit(feats_dem1)

        self.nn_dem0_labels = np.array(labels_dem0)
        self.nn_dem0_preds = np.array(preds_dem0)
        self.nn_dem1_labels = np.array(labels_dem1)
        self.nn_dem1_preds = np.array(preds_dem1)

    # Evaluation with fairness-aware deferral
    def test(self, dataloader, fairness_cost=None):
        if fairness_cost is None:
            fairness_cost = self.fairness_cost
        defers, preds_all, labels_all, hum_preds_all = [], [], [], []
        probs_all, demos_all, comb_probs_all, comb_preds_all = [], [], [], []

        self.model_class.eval()
        with torch.no_grad():
            for x, y, h, d in dataloader:
                x, y, h, d = x.to(self.device), y.to(self.device), h.to(self.device), d.to(self.device)

                probs = F.softmax(self.model_class(x), dim=1)
                _, preds = torch.max(probs, 1)

                comb_probs, _ = self.combiner.combine_proba(
                    probs.cpu().numpy(),
                    h.cpu().numpy(),
                    y.cpu().numpy()
                )

                x_np = x.cpu().numpy()
                _, idx0_batch = self.nn_dem0.kneighbors(x_np)
                _, idx1_batch = self.nn_dem1.kneighbors(x_np)

                for i in range(x.size(0)):
                    acc0 = np.mean(self.nn_dem0_preds[idx0_batch[i]] == self.nn_dem0_labels[idx0_batch[i]])
                    acc1 = np.mean(self.nn_dem1_preds[idx1_batch[i]] == self.nn_dem1_labels[idx1_batch[i]])

                    parity = abs(acc0 - acc1)

                    if self.human_cost <= fairness_cost * parity:
                        defers.append(1)
                        comb_preds_all.append(h[i].item())
                        comb_probs_all.append(comb_probs[i])
                    else:
                        defers.append(0)
                        comb_preds_all.append(preds[i].item())
                        comb_probs_all.append(probs[i].cpu().numpy())

                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(y.cpu().numpy())
                hum_preds_all.extend(h.cpu().numpy())
                probs_all.extend(probs.cpu().numpy())
                demos_all.extend(d.cpu().numpy())


        return {
            "defers": np.array(defers),
            "labels": np.array(labels_all),
            "preds": np.array(preds_all),
            "hum_preds": np.array(hum_preds_all),
            "class_probs": np.array(probs_all),
            "demographics": np.array(demos_all),
            "combined_probs": np.array(comb_probs_all),
            "combined_preds": np.array(comb_preds_all),
        }
    

    # Full training pipeline

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        verbose=True,
        test_interval=5,
        scheduler=None,
    ):


        optimizer_class = optimizer(self.model_class.parameters(), lr=lr)

        # Train classifier for all epochs
        for epoch in tqdm(range(epochs)):
            self.fit_epoch_class(dataloader_train, optimizer_class, verbose, epoch)

            if scheduler:
                scheduler.step()

        # Fit combiner AFTER classifier training
        self.fit_combiner(dataloader_train)

        # Now safe to evaluate on validation set
        if dataloader_val is not None and verbose:
            metrics_val = compute_classification_metrics(self.test(dataloader_val))
            logging.info(metrics_val)

        # Return test metrics
        return compute_classification_metrics(self.test(dataloader_test))
        # optimizer_class = optimizer(self.model_class.parameters(), lr=lr)

        # for epoch in tqdm(range(epochs)):
        #     self.fit_epoch_class(dataloader_train, optimizer_class, verbose, epoch)

        #     if verbose and epoch % test_interval == 0:
        #         logging.info(
        #             compute_classification_metrics(self.test(dataloader_val))
        #         )

        #     if scheduler:
        #         scheduler.step()

        # self.fit_combiner(dataloader_train)