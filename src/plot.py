# src/plot.py

import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "../logs/training_history.csv"

FIGURE_DIR = "../figures"

os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)


# ============================================================
# Plot One Model
# ============================================================

def plot_model(
    df,
    model_name,
):

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    model_df = df[
        df["model"] == model_name
    ].copy()

    if len(model_df) == 0:

        print(
            f"Warning: "
            f"No data found for {model_name}"
        )

        return

    # --------------------------------------------------------
    # Sort by epoch
    # --------------------------------------------------------

    model_df = model_df.sort_values(
        "epoch"
    )

    epochs = model_df["epoch"]

    # ========================================================
    # Figure 1: Loss
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        epochs,
        model_df["train_loss"],
        color="blue",
        label="Train",
        linewidth=2,
    )

    plt.plot(
        epochs,
        model_df["val_loss"],
        color="red",
        label="Validation",
        linewidth=2,
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        f"{model_name} - Loss"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    loss_path = os.path.join(
        FIGURE_DIR,
        f"{model_name}_loss.png",
    )

    plt.savefig(
        loss_path,
        dpi=300,
    )

    plt.close()

    print(
        f"Saved: {loss_path}"
    )

    # ========================================================
    # Figure 2: Accuracy
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        epochs,
        model_df["train_acc"] * 100,
        color="blue",
        label="Train",
        linewidth=2,
    )

    plt.plot(
        epochs,
        model_df["val_acc"] * 100,
        color="red",
        label="Validation",
        linewidth=2,
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        f"{model_name} - Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    accuracy_path = os.path.join(
        FIGURE_DIR,
        f"{model_name}_accuracy.png",
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
    )

    plt.close()

    print(
        f"Saved: {accuracy_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Loaded {len(df)} records."
    )

    print(
        f"Models: "
        f"{df['model'].unique()}"
    )

    # --------------------------------------------------------
    # CNN Baseline
    # --------------------------------------------------------

    plot_model(
        df,
        "CNNBaseline",
    )

    # --------------------------------------------------------
    # ResNet-18
    # --------------------------------------------------------

    plot_model(
        df,
        "ResNet18",
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()