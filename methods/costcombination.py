import time
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append("..")

from helpers.utils import AverageMeter, accuracy
from helpers.metrics import compute_classification_metrics, compute_deferral_metrics
from baselines.basemethod import BaseMethod
from methods.oraclecombiner import OracleCombiner

eps_cst = 1e-8

class PL_Combine_Cost(BaseMethod):
    """
    Cost-sensitive P+L combination using an oracle combiner.

    The system chooses between model and human predictions
    to minimize expected cost.
    """

    def __init__(
        self,
        model_class,
        device,
        miss_cost=9.0,
        human_cost=1.0,
        plotting_interval=100,
    ):
        self.model_class = model_class
        self.device = device
        self.plotting_interval = plotting_interval

        self.miss_cost = miss_cost
        self.human_cost = human_cost
        self.combiner = OracleCombiner()

    # Train classifier (no deferral learning)
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
                logging.warning("NaN loss encountered")
                break

            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    f"Epoch [{epoch}][{batch}/{len(dataloader)}] "
                    f"Loss {losses.avg:.4f} | Acc {top1.avg:.2f}"
                )

    # Fit oracle combiner (uses true labels!)
    def fit_combiner(self, dataloader):
        truths, hum_preds, class_probs = [], [], []

        self.model_class.eval()
        with torch.no_grad():
            for data_x, data_y, h_preds, demographics in dataloader:
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                h_preds = h_preds.to(self.device)

                probs = F.softmax(self.model_class(data_x), dim=1)

                truths.extend(data_y.cpu().numpy())
                hum_preds.extend(h_preds.cpu().numpy())
                class_probs.extend(probs.cpu().numpy())

        self.combiner.fit(
            np.array(class_probs),
            np.array(hum_preds),
            np.array(truths),
        )


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

            if verbose and epoch % test_interval == 0:
                 # Evaluate on validation set
                 val_res = self.test(dataloader_val)
                 logging.info(f"Epoch {epoch}: Val Acc {compute_deferral_metrics(val_res).get('classifier_all_acc', 'N/A')}")


            if scheduler:
                scheduler.step()

        # Fit oracle AFTER classifier training
        self.fit_combiner(dataloader_train)

        return compute_deferral_metrics(self.test(dataloader_test))

    # Evaluation

    def test(self, dataloader):
        defers, truths, hum_preds, preds = [], [], [], []
        class_probs, demographics = [], []
        combined_probs, combined_preds = [], []

        self.model_class.eval()
        with torch.no_grad():
            for data_x, data_y, h_preds, demo in dataloader:
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                h_preds = h_preds.to(self.device)
                demo = demo.to(self.device)

                probs = F.softmax(self.model_class(data_x), dim=1)
                _, model_preds = torch.max(probs, 1)

                comb_probs, defer, _ = self.combiner.combine_proba(
                    probs.cpu().numpy(),
                    h_preds.cpu().numpy(),
                    data_y.cpu().numpy(),
                    miss_cost=self.miss_cost,
                    human_cost=self.human_cost,
                )
                comb_preds, _ = self.combiner.combine(
                    probs.cpu().numpy(),
                    h_preds.cpu().numpy(),
                    data_y.cpu().numpy(),
                    miss_cost=self.miss_cost,
                    human_cost=self.human_cost,
                )

                preds.extend(model_preds.cpu().numpy())
                truths.extend(data_y.cpu().numpy())
                hum_preds.extend(h_preds.cpu().numpy())
                class_probs.extend(probs.cpu().numpy())
                demographics.extend(demo.cpu().numpy())
                combined_probs.extend(comb_probs)
                combined_preds.extend(comb_preds)
                defers.extend(defer)

        return {
            "defers": np.array(defers),
            "labels": np.array(truths),
            "hum_preds": np.array(hum_preds),
            "preds": np.array(preds),
            "class_probs": np.array(class_probs),
            "demographics": np.array(demographics),
            "combined_probs": np.array(combined_probs),
            "combined_preds": np.array(combined_preds),
        }