import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import ResNet18

SEED = 266978

BATCH_SIZE = 128
EPOCHS = 50

LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

DATA_DIR = "./data/cifar-10-batches-py"
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "resnet18_best.pt",
)
os.makedirs(CHECKPOINT_DIR, exist_ok = True)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  
def get_device():
  if torch.backends.mps.is_available():
    return torch.device("mps")
  if torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")
  
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
    
    logits = model(x)
    loss = criterion(logits,y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item() * x.size(0)
    predictions = logits.argmax(dim=1)
    correct +=(predictions == y).sum().item()
    
    total += y.size(0)
    
  epoch_loss = total_loss / total
  epoch_acc = correct / total
  return epoch_loss, epoch_acc

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
    
    logits = model(x)
    loss = criterion(logits, y)
    
    total_loss += loss.item() * x.size(0)
    predictions = logits.argmax(dim=1)
    correct += (predictions == y).sum().item()
    total += y.size(0)
  
  epoch_loss = total_loss / total
  epoch_acc = correct / total
  
  return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, path):
  checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_acc": best_val_acc,
    "seed": SEED,
  }
  torch.save(
    checkpoint,
    path,
  )

def main():
  set_seed(SEED)
  
  device = get_device()
  print(f"Device:{device}")
  if device.type == "mps":
        print("Using Apple Silicon GPU through MPS.")
        
  train_loader, val_loader, test_loader = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    val_ratio=0.1,
    seed=SEED,
    num_workers=0,
  )
  print(f"Train batches: {len(train_loader)}")
  print(f"Val batches:   {len(val_loader)}")
  print(f"Test batches:  {len(test_loader)}")
  
  model = ResNet18(num_classes = 10)
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
  
  history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "lr": [],
    "epoch_time": [],
  }
  
  best_val_acc = 0.0
  total_star_time = time.time()
  print("\nStarting training...\n")
  
  for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    train_loss, train_acc = train_one_epoch(
      model = model,
      loader = train_loader,
      criterion = criterion,
      optimizer = optimizer,
      device = device,
    )    
    
    val_loss, val_acc = evaluate(
      model = model,
      loader = val_loader,
      criterion = criterion,
      device = device,
    )
    
    scheduler.step()
    
    current_lr = optimizer.param_groups[0]["lr"]
    epoch_time = time.time() - epoch_start_time
    
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)
    history["epoch_time"].append(epoch_time)
    
    print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f""
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f""
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f""
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )
    
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch+1,
        best_val_acc=best_val_acc,
        path=CHECKPOINT_PATH,
      )
      print(
        f"  -> Saved best checkpoint "
        f"(Val Acc: {best_val_acc * 100:.2f}%)"
      )

    total_time = time.time() - total_star_time
    
    print("\nLoading best checkpoint...")
    
    checkpoint = torch.load(
      CHECKPOINT_PATH,
      map_location = device,
    )
    model.load_state_dict(
      checkpoint["model_state_dict"]
    )
    best_val_acc = checkpoint["best_val_acc"]
    
    test_loss, test_acc = evaluate(
      model=model,
      loader=test_loader,
      criterion=criterion,
      device=device,
    )
    
    print("\n==============================")
    print("Final Results")
    print("==============================")
    print(f"Model: ResNet-18")
    print(
        f"Best Val Accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )
    print(
        f"Test Accuracy: "
        f"{test_acc * 100:.2f}%"
    )
    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )
    print(
        f"Total Training Time: "
        f"{total_time / 60:.2f} min"
    )
    print(
        f"Checkpoint: ",
        f"{CHECKPOINT_PATH}"
    )
    
if __name__ == "__main__":
  main()
    