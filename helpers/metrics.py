import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import copy


def compute_deferral_metrics(data_test):
    """
    Compute metrics for a human–AI deferral system.

    Args:
        data_test (dict): Must contain
            - 'preds': model predictions
            - 'hum_preds': human predictions
            - 'labels': ground truth labels
            - 'defers': deferral indicators (1 = defer to human)

    Returns:
        dict: deferral and accuracy metrics
    """
    results = {}

    # Overall classifier accuracy
    if len(data_test["labels"]) > 0:
        results["classifier_all_acc"] = accuracy_score(
            data_test["preds"], data_test["labels"]
        )
        results["human_all_acc"] = accuracy_score(
            data_test["hum_preds"], data_test["labels"]
        )
    else:
        results["classifier_all_acc"] = np.nan
        results["human_all_acc"] = np.nan

    # Coverage (fraction of samples NOT deferred)
    if len(data_test["defers"]) > 0:
        results["coverage"] = 1 - np.mean(data_test["defers"])
    else:
        results["coverage"] = np.nan

    # Classifier accuracy on non-deferred samples
    mask_model = data_test["defers"] == 0
    if np.sum(mask_model) > 0:
        results["classifier_nondeferred_acc"] = accuracy_score(
            data_test["preds"][mask_model],
            data_test["labels"][mask_model],
        )
    else:
        results["classifier_nondeferred_acc"] = np.nan

    
    # Human accuracy on deferred samples
    mask_human = data_test["defers"] == 1
    if np.sum(mask_human) > 0:
        results["human_deferred_acc"] = accuracy_score(
            data_test["hum_preds"][mask_human],
            data_test["labels"][mask_human],
        )
    else:
        results["human_deferred_acc"] = np.nan

    # System accuracy (model OR human)
    if len(data_test["labels"]) > 0:
        system_preds = (
            data_test["preds"] * (1 - data_test["defers"])
            + data_test["hum_preds"] * data_test["defers"]
        )
        results["system_acc"] = accuracy_score(system_preds, data_test["labels"])
    else:
        results["system_acc"] = np.nan

    return results

    # results["classifier_all_acc"] = accuracy_score(
    #     data_test["preds"], data_test["labels"]
    # )

    # results["human_all_acc"] = accuracy_score(
    #     data_test["hum_preds"], data_test["labels"]
    # )

    # results["coverage"] = 1 - np.mean(data_test["defers"])

    # # Classifier accuracy when NOT deferred
    # mask_model = data_test["defers"] == 0
    # results["classifier_nondeferred_acc"] = accuracy_score(
    #     data_test["preds"][mask_model],
    #     data_test["labels"][mask_model],
    # )

    # # Human accuracy when deferred
    # mask_human = data_test["defers"] == 1
    # results["human_deferred_acc"] = accuracy_score(
    #     data_test["hum_preds"][mask_human],
    #     data_test["labels"][mask_human],
    # )

    # # System accuracy (model OR human)
    # system_preds = (
    #     data_test["preds"] * (1 - data_test["defers"])
    #     + data_test["hum_preds"] * data_test["defers"]
    # )
    # results["system_acc"] = accuracy_score(
    #     system_preds, data_test["labels"]
    # )

    # return results

def compute_classification_metrics(data_test):
    """
    Compute metrics for classifier-only evaluation.

    Args:
        data_test (dict): Must contain
            - 'preds'
            - 'labels'
            Optional:
            - 'preds_proba'

    Returns:
        dict: classification metrics
    """
    results = {}
    if len(data_test["labels"]) == 0:
        results["classifier_all_acc"] = np.nan
        return results

    results["classifier_all_acc"] = accuracy_score(
        data_test["preds"], data_test["labels"]
    )

    # Binary classification metrics
    if (
        len(np.unique(data_test["labels"])) == 2
        and len(np.unique(data_test["preds"])) == 2
    ):
        results["classifier_all_f1"] = f1_score(
            data_test["labels"], data_test["preds"]
        )

        if "preds_proba" in data_test:
            results["auc"] = roc_auc_score(
                data_test["labels"], data_test["preds_proba"]
            )
        else:
            results["auc"] = np.nan

    return results

def compute_coverage_v_acc_curve(data_test):
    """
    Compute deferral metrics over different coverage levels.

    Args:
        data_test (dict): Must contain
            - 'defers'
            - 'labels'
            - 'hum_preds'
            - 'preds'
            - 'rej_score'

    Returns:
        list: deferral metrics at multiple coverage thresholds
    """
    rej_scores = np.unique(data_test["rej_score"])
    quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))

    all_metrics = [compute_deferral_metrics(data_test)]

    for q in quantiles:
        defers = (data_test["rej_score"] > q).astype(int)

        copy_data = copy.deepcopy(data_test)
        copy_data["defers"] = defers

        all_metrics.append(compute_deferral_metrics(copy_data))

    return all_metrics

def accuracy_gap_per_group(data, use_combined=True):
    """
    Computes accuracy gap across demographic groups.

    Args:
        data (dict): output of test()
        use_combined (bool): whether to use system predictions

    Returns:
        dict with per-group accuracy and gap
    """
    demographics = data["demographics"]
    labels = data["labels"]

    if use_combined:
        preds = (
            data["preds"] * (1 - data["defers"])
            + data["hum_preds"] * data["defers"]
        )
    else:
        preds = data["preds"]

    results = {}
    accuracies = {}

    for g in np.unique(demographics):
        mask = demographics == g
        if mask.sum() == 0:
            acc = np.nan
        else:
            acc = np.mean(preds[mask] == labels[mask])
        accuracies[g] = acc
        results[f"accuracy_group_{g}"] = acc

    # Accuracy gap (only if 2 groups)
    if len(accuracies) == 2:
        groups = list(accuracies.keys())
        results["accuracy_gap"] = abs(
            accuracies[groups[0]] - accuracies[groups[1]]
        )
    else:
        results["accuracy_gap"] = np.nan

    return results


