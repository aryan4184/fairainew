import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix

def compute_bound_components(y_true, y_pred, sensitive_features, positive_class=1):
    """
    Computes the components of the accuracy bound:
    LHS = |A_0 - A_1| (Accuracy gap)
    RHS = 2 * max(pi_max, 1-pi_min) * max(|Delta TPR|, |Delta FPR|)
    
    Returns dictionary with components.
    """
    # Convert to binary for the specific class of interest if multiclass
    # But Adult is binary relative to "positive_class" (usually >50k)
    # The code uses indices 0, 1, 2... so we might need to be careful.
    # Assuming input is already binary or we simply compare y_pred == y_true.
    
    # Identify groups
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, found {len(groups)}")
    
    g0 = groups[0]
    g1 = groups[1]
    
    mask0 = sensitive_features == g0
    mask1 = sensitive_features == g1
    
    # Compute Accuracies
    acc0 = np.mean(y_pred[mask0] == y_true[mask0])
    acc1 = np.mean(y_pred[mask1] == y_true[mask1])
    lhs = abs(acc0 - acc1)
    
    # Compute Base Rates (pi)
    # pi_g = P(Y=1 | G=g)
    pi0 = np.mean(y_true[mask0] == positive_class)
    pi1 = np.mean(y_true[mask1] == positive_class)
    
    pi_max = max(pi0, pi1)
    # pi_min = min(pi0, pi1)
    pi_min = min(pi0, pi1) 
    # P(Y=0) = 1 - P(Y=1)

    scaling_factor = max(pi_max, 1 - pi_min) 


    
    def get_tpr_fpr(y_t, y_p, pos_label):

        yt_bin = (y_t == pos_label).astype(int)
        yp_bin = (y_p == pos_label).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(yt_bin, yp_bin, labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return tpr, fpr

    tpr0, fpr0 = get_tpr_fpr(y_true[mask0], y_pred[mask0], positive_class)
    tpr1, fpr1 = get_tpr_fpr(y_true[mask1], y_pred[mask1], positive_class)
    
    delta_tpr = abs(tpr0 - tpr1)
    delta_fpr = abs(fpr0 - fpr1)
    
    max_delta = max(delta_tpr, delta_fpr)
    
    rhs = 2 * scaling_factor * max_delta
    
    return {
        "lhs": float(lhs),
        "rhs": float(rhs),
        "tightness": float(lhs / rhs) if rhs > 0 else 0.0,
        "holds": bool(lhs <= rhs + 1e-9), # tolerance
        "acc0": float(acc0),
        "acc1": float(acc1),
        "tpr0": float(tpr0),
        "tpr1": float(tpr1),
        "fpr0": float(fpr0),
        "fpr1": float(fpr1),
        "delta_tpr": float(delta_tpr),
        "delta_fpr": float(delta_fpr),
        "pi0": float(pi0),
        "pi1": float(pi1)
    }

def bootstrap_metrics(y_true, y_pred, sensitive_features, n_bootstrap=200, positive_class=1):
    """
    Bootstrap resampling to estimate stability of the bound.
    Returns means and stds.
    """
    n = len(y_true)
    results = []
    violations = 0
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        yt_b = y_true[indices]
        yp_b = y_pred[indices]
        sf_b = sensitive_features[indices]
        
        # Skip if one group is missing in bootstrap sample (unlikely but possible)
        if len(np.unique(sf_b)) < 2:
            continue
            
        res = compute_bound_components(yt_b, yp_b, sf_b, positive_class)
        results.append(res)
        if not res["holds"]:
            violations += 1
            
    # Aggregate
    keys = results[0].keys()
    aggregated = {}
    for k in keys:
        values = [r[k] for r in results]
        aggregated[f"{k}_mean"] = np.mean(values)
        aggregated[f"{k}_std"] = np.std(values)
        
    aggregated["violation_rate"] = violations / len(results)
    return aggregated

def statistical_tests(config_results):
    """
    Hypothesis testing.
    config_results: list of dicts with 'tightness', 'mode', etc. for different runs.
    
    We want to test if 'Fair' mode has different tightness than 'Cost'.
    """
    
    pass
