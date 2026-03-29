import time
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score

sys.path.append("..")

from helpers.utils import AverageMeter, accuracy
from baselines.basemethod import BaseMethod

eps_cst = 1e-8


class SelectivePredictionSep(BaseMethod):
    """
    Selective Prediction with separate confidence thresholds
    for different demographic groups.
    """

    def __init__(self, model_class, device, plotting_interval=100):
        self.model_class = model_class
        self.device = device
        self.plotting_interval = plotting_interval

        # Separate thresholds for each demographic group
        self.demographic0 = 0.5
        self.demographic1 = 0.5

    # Train classifier (no deferral learning here)
    def fit_epoch_class(self, dataloader, optimizer, verbose=True, epoch=1):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        self.model_class.train()
        for batch, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)

            outputs = self.model_class(data_x)
            loss = F.cross_entropy(outputs, data_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if torch.isnan(loss):
                logging.warning("NaN loss encountered, stopping training")
                break

    # Learn demographic-specific thresholds on validation
    def set_optimal_threshold(self, dataloader):
        """
        Learn separate confidence thresholds for each demographic
        by maximizing system accuracy on the validation set.
        """
        data_preds = self.test(dataloader)

        threshold_grid = np.concatenate([
            data_preds["max_probs"],
            np.linspace(0, 1, 20)
        ])

        best_thr_0, best_acc_0 = 0.0, 0.0
        best_thr_1, best_acc_1 = 0.0, 0.0

        mask = data_preds["demographics"].astype(bool)

        group0 = {k: v[~mask] for k, v in data_preds.items()}
        group1 = {k: v[mask] for k, v in data_preds.items()}

        # Optimize threshold for demographic = 0
        for t in threshold_grid:
            defers = (group0["max_probs"] < t).astype(int)
            acc = accuracy_score(
                group0["preds"] * (1 - defers) + group0["hum_preds"] * defers,
                group0["labels"],
            )
            if acc > best_acc_0:
                best_acc_0, best_thr_0 = acc, t

        # Optimize threshold for demographic = 1
        for t in threshold_grid:
            defers = (group1["max_probs"] < t).astype(int)
            acc = accuracy_score(
                group1["preds"] * (1 - defers) + group1["hum_preds"] * defers,
                group1["labels"],
            )
            if acc > best_acc_1:
                best_acc_1, best_thr_1 = acc, t

        self.demographic0 = best_thr_0
        self.demographic1 = best_thr_1

        logging.info(f"Best threshold (group 0): {best_thr_0}, acc={best_acc_0}")
        logging.info(f"Best threshold (group 1): {best_thr_1}, acc={best_acc_1}")

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
        if scheduler:
            scheduler = scheduler(optimizer_class, len(dataloader_train) * epochs)

        for epoch in tqdm(range(epochs)):
            self.fit_epoch_class(dataloader_train, optimizer_class, verbose, epoch)
            if scheduler:
                scheduler.step()


        # Learn thresholds AFTER classifier training
        self.set_optimal_threshold(dataloader_val)

    # Evaluation
    def test(self, dataloader):
        defers_all, max_probs = [], []
        truths_all, hum_preds_all, predictions_all = [], [], []
        rej_score_all, class_probs_all, demographics_all = [], [], []

        self.model_class.eval()
        with torch.no_grad():
            for data_x, data_y, hum_preds, demographics in dataloader:
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                demographics = demographics.to(self.device)

                probs = F.softmax(self.model_class(data_x), dim=1)
                max_class_probs, preds = torch.max(probs, 1)
            
                for i in range(len(data_y)):
                    thr = self.demographic0 if demographics[i] == 0 else self.demographic1
                    defers_all.append(int(max_class_probs[i] < thr))
                    rej_score_all.append(1 - max_class_probs[i].item())

                max_probs.extend(max_class_probs.cpu().numpy())
                predictions_all.extend(preds.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(probs.cpu().numpy())
                demographics_all.extend(demographics.cpu().numpy())

        return {
            "defers": np.array(defers_all),
            "labels": np.array(truths_all),
            "max_probs": np.array(max_probs),
            "hum_preds": np.array(hum_preds_all),
            "preds": np.array(predictions_all),
            "rej_score": np.array(rej_score_all),
            "class_probs": np.array(class_probs_all),
            "demographics": np.array(demographics_all),
        }