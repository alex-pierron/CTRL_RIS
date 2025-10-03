"""
TensorBoard Data Visualization Script

This script processes TensorBoard log data to generate customized plots for specified metrics.
It loads scalar data from TensorBoard event files, organizes it into a pandas DataFrame,
and creates visualizations using seaborn and matplotlib.

Requirements:
    - TensorBoard event files must be in subdirectories of ROOT_DIR
    - Each subdirectory should contain an 'events.out' file
    - Directory structure and file naming conventions must match expectations

Configuration:
    - ROOT_DIR: Path to directory containing TensorBoard logs
    - OUTPUT_DIR: Path where generated plots will be saved
    - FEATURES: List of metrics to plot
    - CUSTOM_NAMES: Mapping of subfolder names to display names
    - PLOT_CONFIG: Customization options for each plot
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

# ==============================
# CONFIG
# ==============================
ROOT_DIR = "data/analytics/paper/Example_3/ddpg"   # Folder containing tensorboard logs
OUTPUT_DIR = "data/analytics/paper/Example_3/plots"  # Directory to save generated plots
OUTPUT_FORMAT = "pdf"  # or "pdf"  # specify the output format

# Features (metrics) to plot - these should match the scalar tags in your TensorBoard data
FEATURES = [
    "Fairness/User local average Fairness",
    "Rewards/Local Average Baseline Reward",
    "Eavesdropper/Local average reward",
    "Rewards/Baseline Instant Reward",
    "Fairness/Instant JFI",
    "Fairness/JFI for best reward obtained",
    "Rewards/Max Instant Reward",
]

# Mapping of subfolder names to more readable display names for the legend
CUSTOM_NAMES = {
    "subfolder1": "Baseline Reward",
    "subfolder2": "QoS Reward",
    "subfolder3": "R_fm Reward",
    "subfolder4": "R_smoothed Reward",
}

# Customization options for each plot including titles, labels, and line widths
PLOT_CONFIG = {
    "Fairness/User local average Fairness": {
        "title": "User JFI Over Time (Local Average)",
        "xlabel": "Training Steps",
        "ylabel": "Jain Fairness Index",
        "linewidth": 1.5
    },
    "Fairness/Instant JFI": {
        "title": "User Instant JFI Over Time",
        "xlabel": "Training Steps",
        "ylabel": "Jain Fairness Index",
        "linewidth": 1.5
    },

    "Rewards/Local Average Baseline Reward": {
        "title": "Baseline Reward Over Time (Local Average) Bps/Hz",
        "xlabel": "Training Steps",
        "ylabel": "Reward in Bps/Hz",
        "linewidth": 1.75
    },

    "Eavesdropper/Local average reward": {
        "title": "Eavesdropper Reward Over Time (Local Average) Bps/Hz",
        "xlabel": "Training Steps",
        "ylabel": "Eavesdropper Reward in Bps/Hz",
        "linewidth": 1.25
    },

    "Rewards/Baseline Instant Reward": {
        "title": "Baseline Reward Bps/Hz",
        "xlabel": "Training Steps",
        "ylabel": "Eavesdropper Reward in Bps/Hz",
        "linewidth": 1.5
    },
    "Fairness/JFI for best reward obtained":{
        "title": "JFI for the best reward obtained",
        "xlabel": "Training Steps",
        "ylabel": "Jain Fairness Index",
        "linewidth": 1.5
    },
    "Rewards/Max Instant Reward": {
        "title": "Max Instant Reward Reached",
        "xlabel": "Training Steps",
        "ylabel": "Reward",
        "linewidth": 1.5}
}

# Optional smoothing windows (per feature). Set to None or 0 to disable smoothing
SMOOTHING_WINDOWS = {
    "Fairness/User local average Fairness": None,
    "Rewards/Local Average Baseline Reward": None,
    "Eavesdropper/Local average reward": None,  # no smoothing
    "Rewards/Baseline Instant Reward": 100,
    "Fairness/Instant JFI": 100,
    "Fairness/JFI for best reward obtained": None, 
    "Rewards/Max Instant Reward": None,
}


def load_tensorboard_scalars(event_file, feature):
    """
    Load scalar data from a TensorBoard event file for a specific feature.
    """
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    if feature not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(feature)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main():
    """
    Main function that orchestrates the data loading and plotting process.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all scalar data into a DataFrame
    all_data = []
    for subfolder in [f.path for f in os.scandir(ROOT_DIR) if f.is_dir()]:
        event_files = [f for f in os.listdir(subfolder) if f.startswith("events.out")]
        if not event_files:
            print(f"No event file found in {subfolder}")
            continue
        event_file = os.path.join(subfolder, event_files[0])
        for feature in FEATURES:
            steps, values = load_tensorboard_scalars(event_file, feature)
            if steps is None:
                print(f"Feature '{feature}' not found in {subfolder}")
                continue
            run_name = CUSTOM_NAMES.get(os.path.basename(subfolder), os.path.basename(subfolder))
            all_data.extend({"step": s, "value": v, "feature": feature, "run": run_name}
                           for s, v in zip(steps, values))

    if not all_data:
        print("No data extracted!")
        return

    df = pd.DataFrame(all_data)

    # Apply smoothing per feature if requested
    smoothed_dfs = []
    for feature in FEATURES:
        window = SMOOTHING_WINDOWS.get(feature, None)
        sub_df = df[df["feature"] == feature].copy()
        if window and window > 1:
            sub_df["value"] = sub_df.groupby("run")["value"].transform(
                lambda x: x.rolling(window, center=True, min_periods=1).mean()
            )
        smoothed_dfs.append(sub_df)

    df = pd.concat(smoothed_dfs, ignore_index=True)

    # Set visualization style
    sns.set_theme(style="whitegrid", palette="colorblind")

    # Plot each feature with custom configuration
    for feature in FEATURES:
        cfg = PLOT_CONFIG.get(feature, {})
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=df[df["feature"] == feature],
            x="step", y="value", hue="run", linewidth=cfg.get("linewidth", 1.5)
        )
        plt.title(cfg.get("title", feature), fontsize=14)
        plt.xlabel(cfg.get("xlabel", "Step"), fontsize=12)
        plt.ylabel(cfg.get("ylabel", feature), fontsize=12)
        plt.legend(title="Run", loc="best", fontsize=10, title_fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"{feature.replace('/', '_')}.{OUTPUT_FORMAT}")
        #save_path = os.path.join(OUTPUT_DIR, f"{feature.replace('/', '_')}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"Plots saved in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
