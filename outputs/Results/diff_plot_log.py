import re
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Log parsing
# =========================
def parse_log(filepath):
    """
    Parses fairness + accuracy metrics from log file.

    Returns:
        dict[str, dict]:
            regime -> {
                fairness_mean,
                fairness_std,
                accuracy_gap_mean,
                accuracy_gap_std
            }
    """
    with open(filepath, "r") as f:
        text = f.read()

    # Split sections: --- Fair combination ---
    sections = re.split(r"---\s*(.*?)\s*---", text)

    results = {}

    for i in range(1, len(sections), 2):
        regime = sections[i].strip()
        block = sections[i + 1]

        # System Equalized Odds (fairness metric)
        eo_match = re.search(
            r"system_equalized_odds_difference_c0:\s*mean=([0-9.eE+-]+),\s*var=([0-9.eE+-]+)",
            block,
        )

        # Accuracy Gap (accuracy metric)
        acc_gap_match = re.search(
            r"accuracy_gap:\s*mean=([0-9.eE+-]+),\s*var=([0-9.eE+-]+)",
            block,
        )

        if eo_match and acc_gap_match:
            eo_mean, eo_var = map(float, eo_match.groups())
            gap_mean, gap_var = map(float, acc_gap_match.groups())

            results[regime] = {
                # Fairness: 1 - EO diff (higher is better)
                "fairness_mean": 1.0 - eo_mean,
                "fairness_std": np.sqrt(eo_var),
                # Accuracy Gap (lower is better, but plotting raw value)
                "acc_gap_mean": gap_mean,
                "acc_gap_std": np.sqrt(gap_var),
            }

    return results


# =========================
# Plotting
# =========================
def plot_curve(ax, data, color, label):
    # Sort data by fairness for cleaner line plots if needed, 
    # but regimes are discrete points usually.
    # We'll just scatter/errorbar them.
    
    x = [v["fairness_mean"] for v in data.values()]
    y = [v["acc_gap_mean"] for v in data.values()]
    xerr = [v["fairness_std"] for v in data.values()]
    yerr = [v["acc_gap_std"] for v in data.values()]
    names = list(data.keys())

    # Plot points with error bars
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        capsize=5,
        color=color,
        label=label,
        markersize=8
    )
    
    # Annotate points
    for i, txt in enumerate(names):
        ax.annotate(txt, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel("Fairness (1 - Equalized Odds Diff)\n(Higher is better)")
    ax.set_ylabel("Accuracy Gap (Group 0 - Group 1)\n(Lower is better)")
    ax.grid(alpha=0.3)


# =========================
# Main
# =========================
def main():
    log_file = "/Users/aryan/Code/FairnessTheises/Code/output1.log" 
    results = parse_log(log_file)

    if not results:
        print("No results found in log file!")
        return

    # Group regimes
    biased = {
        k: v for k, v in results.items()
        if k in ["Fair", "Cost optimized"]
    }

    unbiased = {
        k: v for k, v in results.items()
        if k in ["All combination", "Fair combination"]
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    if biased:
        plot_curve(
            ax,
            biased,
            color="red",
            label="Biased Ground Truth (Cost / Fair)",
        )
    
    if unbiased:
        plot_curve(
            ax,
            unbiased,
            color="blue",
            label="Unbiased Ground Truth (All / Fair Comb)",
        )

    ax.set_title("Accuracy Difference vs Fairness")
    ax.legend()

    plt.tight_layout()
    output_path = "diff_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
