import sys
import logging
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append("..") 


from helpers.utils import *
from helpers.metrics import *
from baselines.basemethod import BaseMethod
from methods.allcombiner import AllCombiner

eps_cst = 1e-8 

class PL_Combine(BaseMethod):
    """
    Selective prediction method:
    - Train classifier on all data
    - Defer to human or combined prediction based on a learned combiner
    """

    def __init__(self, model_class, device, plotting_interval=100):
        self.model_class = model_class
        self.device = device
        self.plotting_interval = plotting_interval
        self.combiner = AllCombiner()  # class that handles combining human + model predictions
        # set_seed(42)


    # Single training epoch
    def fit_epoch_class(self, dataloader, optimizer, verbose=True, epoch=1):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        self.model_class.train()
        for batch_idx, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
            data_x, data_y = data_x.to(self.device), data_y.to(self.device)

            outputs = self.model_class(data_x)
            loss = F.cross_entropy(outputs, data_y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute accuracy
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))

            # batch time logging
            batch_time.update(time.time() - end)
            end = time.time()

            if torch.isnan(loss):
                logging.warning("NaN loss encountered, stopping training")
                break

            if verbose and batch_idx % self.plotting_interval == 0:
                logging.info(
                    f"Epoch: [{epoch}][{batch_idx}/{len(dataloader)}] "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})"
                )

    # Fit the combiner that decides when to defer
    def fit_combiner(self, dataloader):
        max_probs, truths_all, hum_preds_all, predictions_all, class_probs_all = [], [], [], [], []

        self.model_class.eval()
        with torch.no_grad():
            for data_x, data_y, hum_preds, demographics in dataloader:
                data_x, data_y, hum_preds = data_x.to(self.device), data_y.to(self.device), hum_preds.to(self.device)

                outputs_class = F.softmax(self.model_class(data_x), dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class, 1)

                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
                max_probs.extend(max_class_probs.cpu().numpy())


        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        max_probs = np.array(max_probs)
        class_probs_all = np.array(class_probs_all)

        # Fit the combiner (human + model)
        self.combiner.fit(class_probs_all, hum_preds_all, truths_all)

    # Full training loop
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

        self.model_class.train()
        for epoch in tqdm(range(epochs)):
            self.fit_epoch_class(dataloader_train, optimizer_class, verbose, epoch)
            if scheduler:
                scheduler.step()

        # Fit the combiner AFTER classifier is fully trained
        self.fit_combiner(dataloader_train)


        metrics = compute_deferral_metrics(self.test(dataloader_val))

        model_acc = metrics.get("classifier_all_acc", np.nan)
        human_acc = metrics.get("human_all_acc", np.nan)
        combined_acc = metrics.get("system_acc", np.nan)

        eod_c0 = metrics.get("system_equalized_odds_difference_c0", np.nan)
        eod_c1 = metrics.get("system_equalized_odds_difference_c1", np.nan)
        eod_c2 = metrics.get("system_equalized_odds_difference_c2", np.nan)

        deferral_rate = metrics.get("deferral_rate", np.nan)

        logging.info(
            f"EVAL_METRICS "
            f"model_acc={model_acc:.4f} "
            f"human_acc={human_acc:.4f} "
            f"combined_acc={combined_acc:.4f} "
            f"eod_c0={eod_c0:.4f} "
            f"eod_c1={eod_c1:.4f} "
            f"eod_c2={eod_c2:.4f} "
            f"deferral_rate={deferral_rate:.4f}"
        )

        # # Run validation/testing
        # if verbose:
        #     for epoch in range(epochs):
        #         if epoch % test_interval == 0:
        #             metrics_val = compute_deferral_metrics(self.test(dataloader_val))

        #             # Safely get metrics
        #             model_acc = metrics_val.get("classifier_all_acc", np.nan)
        #             human_acc = metrics_val.get("human_all_acc", np.nan)
        #             combined_acc = metrics_val.get("system_acc", np.nan)

        #             eod_c0 = metrics_val.get("system_equalized_odds_difference_c0", np.nan)
        #             eod_c1 = metrics_val.get("system_equalized_odds_difference_c1", np.nan)
        #             eod_c2 = metrics_val.get("system_equalized_odds_difference_c2", np.nan)

        #             logging.info(
        #                 f"EPOCH_METRICS "
        #                 f"epoch={epoch} "
        #                 f"model_acc={model_acc:.4f} "
        #                 f"human_acc={human_acc:.4f} "
        #                 f"combined_acc={combined_acc:.4f} "
        #                 f"eod_c0={eod_c0} "
        #                 f"eod_c1={eod_c1} "
        #                 f"eod_c2={eod_c2}"
        #             )

                # if epoch % test_interval == 0:
                #     metrics_val = compute_classification_metrics(self.test(dataloader_val))
                #     logging.info(metrics_val)
            # if epoch == 0:
            #     self.fit_combiner(dataloader_train)


            # if verbose and epoch % test_interval == 0:
            #     metrics_val = compute_classification_metrics(self.test(dataloader_val))
            #     logging.info(metrics_val)

            # if scheduler:
            #     scheduler.step()

        # Fit combiner after classifier is trained
        # self.fit_combiner(dataloader_train)

        return compute_deferral_metrics(self.test(dataloader_test))
    
    # Evaluation
    def test(self, dataloader):
        truths_all, hum_preds_all, predictions_all = [], [], []
        class_probs_all, combined_probs_all, combined_preds_all = [], [], []
        defers_all, max_probs, demographics_all = [], [], []

        self.model_class.eval()
        with torch.no_grad():
            for data_x, data_y, hum_preds, demographics in dataloader:
                data_x, data_y, hum_preds, demographics = (
                    data_x.to(self.device),
                    data_y.to(self.device),
                    hum_preds.to(self.device),
                    demographics.to(self.device),
                )

                outputs_class = F.softmax(self.model_class(data_x), dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class, 1)

                # Combined prediction using learned combiner
                combined_probs, _ = self.combiner.combine_proba(outputs_class.cpu().numpy(),
                                                             hum_preds.cpu().numpy(),
                                                             data_y.cpu().numpy())
                combined_preds, _ = self.combiner.combine(outputs_class.cpu().numpy(),
                                                          hum_preds.cpu().numpy(),
                                                          data_y.cpu().numpy())
                
                # Store all outputs
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                predictions_all.extend(predicted_class.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())

                for p in combined_probs:
                    combined_probs_all.append(np.asarray(p).reshape(-1))
                # combined_probs_all.extend(combined_probs)

                combined_preds_all.extend(combined_preds)
                max_probs.extend(max_class_probs.cpu().numpy())
                demographics_all.extend(demographics.cpu().numpy())
                defers_all.extend(np.ones(len(data_y)))

        # # --- DEBUG CHECKS ---
        # print("\nDEBUG shapes:")
        # print("preds:", np.array(predictions_all).shape)
        # print("defers:", np.array(defers_all).shape)
        # print("combined_preds:", np.array(combined_preds_all).shape)
        # print("labels:", np.array(truths_all).shape)

        # print("\nDEBUG sample values (first 10):")
        # print("preds[:10]:", predictions_all[:10])
        # print("defers[:10]:", defers_all[:10])
        # print("combined_preds[:10]:", combined_preds_all[:10])
        # print("labels[:10]:", truths_all[:10])

        # Check if all predictions were deferred



        # Prepare output dictionary
        data = {
            "defers": np.array(defers_all),
            "labels": np.array(truths_all),
            "max_probs": np.array(max_probs),
            "hum_preds": np.array(hum_preds_all),
            "preds": np.array(predictions_all),
            "class_probs": np.array(class_probs_all),
            "demographics": np.array(demographics_all),
            # "combined_probs": np.array(combined_probs_all),
            "combined_probs": np.stack(combined_probs_all, axis=0),
            "combined_preds": np.array(combined_preds_all),
        }
        return data