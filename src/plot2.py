# src/plot2.py

import os

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "../logs/training_history.csv"

FIGURE_DIR = "../figures"

os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)

def main():

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    print(
        f"Loaded {len(df)} records."
    )

    # ========================================================
    # Separate models
    # ========================================================

    baseline_df = df[
        df["model"] == "CNNBaseline"
    ].sort_values("epoch")

    resnet_df = df[
        df["model"] == "ResNet18"
    ].sort_values("epoch")

    # ========================================================
    # Figure 1: Loss
    # ========================================================

    plt.figure(figsize=(9, 6))

    # Baseline - Train
    plt.plot(
        baseline_df["epoch"],
        baseline_df["train_loss"],
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Baseline Train",
    )

    # Baseline - Validation
    plt.plot(
        baseline_df["epoch"],
        baseline_df["val_loss"],
        color="red",
        linestyle="-",
        linewidth=2,
        label="Baseline Val",
    )

    # ResNet18 - Train
    plt.plot(
        resnet_df["epoch"],
        resnet_df["train_loss"],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="ResNet18 Train",
    )

    # ResNet18 - Validation
    plt.plot(
        resnet_df["epoch"],
        resnet_df["val_loss"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="ResNet18 Val",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    loss_path = os.path.join(
        FIGURE_DIR,
        "comparison_loss.png",
    )

    plt.savefig(
        loss_path,
        dpi=300,
    )

    print(
        f"Saved: {loss_path}"
    )

    # ========================================================
    # Figure 2: Accuracy
    # ========================================================

    plt.figure(figsize=(9, 6))

    # Baseline - Train
    plt.plot(
        baseline_df["epoch"],
        baseline_df["train_acc"] * 100,
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Baseline Train",
    )

    # Baseline - Validation
    plt.plot(
        baseline_df["epoch"],
        baseline_df["val_acc"] * 100,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Baseline Val",
    )

    # ResNet18 - Train
    plt.plot(
        resnet_df["epoch"],
        resnet_df["train_acc"] * 100,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="ResNet18 Train",
    )

    # ResNet18 - Validation
    plt.plot(
        resnet_df["epoch"],
        resnet_df["val_acc"] * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        label="ResNet18 Val",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    accuracy_path = os.path.join(
        FIGURE_DIR,
        "comparison_accuracy.png",
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
    )

    print(
        f"Saved: {accuracy_path}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()