# src/train.py

import os
import random
import time
import csv

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import CNNBaseline, ResNet18


# ============================================================
# Configuration
# ============================================================

SEED = 266978

BATCH_SIZE = 128
EPOCHS = 200

LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

DATA_DIR = "../data/cifar-10-batches-py"

CHECKPOINT_DIR = "../checkpoints"
LOG_DIR = "../logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# CSV 文件
CSV_PATH = os.path.join(
    LOG_DIR,
    "training_history.csv",
)


# ============================================================
# Random Seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        # forward
        logits = model(x)
        loss = criterion(logits, y)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += (loss.item() * batch_size)
        predictions = logits.argmax(dim=1)
        correct += ((predictions == y).sum().item())
        total += batch_size

    epoch_loss = total_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ============================================================
# Validation / Test
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # forward
        logits = model(x)
        loss = criterion(logits, y)

        batch_size = x.size(0)
        total_loss += (loss.item() * batch_size)
        predictions = logits.argmax(dim=1)
        correct += ((predictions == y).sum().item())
        total += batch_size

    epoch_loss = total_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ============================================================
# Save Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_acc,
    seed,
    path,
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "seed": seed,
    }

    torch.save(checkpoint, path)


# ============================================================
# Create CSV
# ============================================================

def create_csv():
  
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
        ])


# ============================================================
# Append One Epoch to CSV
# ============================================================

def log_to_csv(
    model_name,
    epoch,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
):

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        ])


# ============================================================
# Train One Model
# ============================================================

def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    device,
):

    print("\n")
    print("=" * 70)
    print(f"Training: {model_name}")
    print("=" * 70)

    model = model.to(device)
    print(model)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print(
        f"\nTrainable parameters: "
        f"{num_parameters:,}"
    )
    print(
        f"Trainable parameters: "
        f"{num_parameters / 1e6:.2f} M"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{model_name}_best.pt",
    )

    best_val_acc = 0.0
    total_start_time = time.time()
    print("\nStarting training...\n")

    # ========================================================
    # Epoch Loop
    # ========================================================

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = (time.time() - epoch_start_time)

        log_to_csv(
            model_name=model_name,
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f"| "
            f"Train Loss: {train_loss:.4f} "
            f"| "
            f"Train Acc: {train_acc * 100:.2f}% "
            f"| "
            f"Val Loss: {val_loss:.4f} "
            f"| "
            f"Val Acc: {val_acc * 100:.2f}% "
            f"| "
            f"LR: {current_lr:.6f} "
            f"| "
            f"Time: {epoch_time:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_val_acc=best_val_acc,
                seed=SEED,
                path=checkpoint_path,
            )
            print(
                f"    -> Best checkpoint saved "
                f"(Val Acc: "
                f"{best_val_acc * 100:.2f}%)"
            )

    total_time = (time.time() - total_start_time)

    print("\nLoading best checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_val_acc = checkpoint["best_val_acc"]
    
    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\n")
    print("-" * 70)
    print(f"Final Results: {model_name}")
    print("-" * 70)

    print(
        f"Best Val Accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )
    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )
    print(
        f"Test Accuracy: "
        f"{test_acc * 100:.2f}%"
    )
    print(
        f"Training Time: "
        f"{total_time / 60:.2f} min"
    )
    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )
    print("-" * 70)

    return {
        "model": model_name,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "training_time": total_time,
        "num_parameters": num_parameters,
    }

def main():

    set_seed(SEED)
    device = get_device()
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "mps":
        print(
            "Using Apple Silicon GPU "
            "through MPS."
        )
    elif device.type == "cuda":
        print(
            f"CUDA GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        print(
            "WARNING: Using CPU."
        )

    print("=" * 70)

    train_loader, val_loader, test_loader = (
        get_dataloaders(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            val_ratio=0.1,
            seed=SEED,
            num_workers=0,
        )
    )

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )
    print(
        f"Val samples:   "
        f"{len(val_loader.dataset)}"
    )
    print(
        f"Test samples:  "
        f"{len(test_loader.dataset)}"
    )
    print(
        f"Train batches: "
        f"{len(train_loader)}"
    )
    print(
        f"Val batches:   "
        f"{len(val_loader)}"
    )
    print(
        f"Test batches:  "
        f"{len(test_loader)}"
    )

    create_csv()
    print(
        f"\nTraining history will be saved to:"
        f"\n{CSV_PATH}"
    )
    
    # CNNBaseline
    set_seed(SEED)
    baseline_model = CNNBaseline(
        num_classes=10
    )
    baseline_result = train_model(
        model=baseline_model,
        model_name="CNNBaseline",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )
    print("\n\n")
    print("#" * 70)
    print("# CNN Baseline finished.")
    print("# Starting ResNet-18...")
    print("#" * 70)

    # ResNet18
    set_seed(SEED)
    resnet_model = ResNet18(
        num_classes=10
    )
    resnet_result = train_model(
        model=resnet_model,
        model_name="ResNet18",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    # Comparison
    print("\n\n")
    print("=" * 70)
    print("Overall Comparison")
    print("=" * 70)
    print(
        f"{'Model':<15}"
        f"{'Best Val Acc':>15}"
        f"{'Test Acc':>15}"
        f"{'Time(min)':>15}"
        f"{'Params(M)':>15}"
    )
    print("-" * 70)
    print(
        f"{baseline_result['model']:<15}"
        f"{baseline_result['best_val_acc'] * 100:>14.2f}%"
        f"{baseline_result['test_acc'] * 100:>14.2f}%"
        f"{baseline_result['training_time'] / 60:>15.2f}"
        f"{baseline_result['num_parameters'] / 1e6:>15.2f}"
    )
    print(
        f"{resnet_result['model']:<15}"
        f"{resnet_result['best_val_acc'] * 100:>14.2f}%"
        f"{resnet_result['test_acc'] * 100:>14.2f}%"
        f"{resnet_result['training_time'] / 60:>15.2f}"
        f"{resnet_result['num_parameters'] / 1e6:>15.2f}"
    )
    print("=" * 70)
    print(
        f"\nTraining history saved to:"
        f"\n{CSV_PATH}"
    )

if __name__ == "__main__":
    main()