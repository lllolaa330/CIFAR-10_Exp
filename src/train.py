"""Configuration-driven training with one reproducible directory per run."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import CNNBaseline, ResNet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_FIELDS = [
    "epoch",
    "global_step",
    "lr",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "epoch_time_s",
    "elapsed_time_s",
    "allocated_memory_mb",
    "peak_memory_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one CIFAR-10 experiment and save it as an isolated run."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="JSON experiment config")
    source.add_argument(
        "--resume",
        type=Path,
        help="Path to an existing run's last_checkpoint.pt",
    )
    parser.add_argument("--seed", type=int, help="Override seed for a new run")
    parser.add_argument("--epochs", type=int, help="Override epochs for a new run")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Override configured device",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not evaluate the test set at the end",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and model without creating a run",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temp_path, path)


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    torch.save(value, temp_path)
    os.replace(temp_path, path)


def torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_config(config: dict[str, Any]) -> None:
    required = ["experiment", "variant", "display_name", "seed"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(missing)}")

    if "model" not in config or "name" not in config["model"]:
        raise ValueError("Config must contain model.name")
    if config["model"]["name"] not in {"CNNBaseline", "ResNet18"}:
        raise ValueError(f"Unsupported model: {config['model']['name']}")

    training = config.get("training", {})
    for key in ("epochs", "batch_size", "criterion", "optimizer", "scheduler"):
        if key not in training:
            raise ValueError(f"Config must contain training.{key}")
    if int(training["epochs"]) <= 0 or int(training["batch_size"]) <= 0:
        raise ValueError("training.epochs and training.batch_size must be positive")

    data = config.get("data", {})
    for key in ("data_dir", "val_ratio", "num_workers", "augmentation"):
        if key not in data:
            raise ValueError(f"Config must contain data.{key}")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_")
    return value.lower() or "run"


def unique_run_dir(config: dict[str, Any]) -> tuple[str, Path]:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    stem = "_".join(
        [
            timestamp,
            slugify(str(config["experiment"])),
            slugify(str(config["variant"])),
            f"s{config['seed']}",
        ]
    )
    runs_dir = PROJECT_ROOT / config.get("output", {}).get("runs_dir", "runs")
    candidate = runs_dir / stem
    suffix = 1
    while candidate.exists():
        candidate = runs_dir / f"{stem}-{suffix:02d}"
        suffix += 1
    return candidate.name, candidate


def git_metadata() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run_git("status", "--porcelain")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "git": git_metadata(),
        "command": sys.argv,
    }


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = dict(config["model"])
    name = model_config.pop("name")
    if name == "CNNBaseline":
        return CNNBaseline(**model_config)
    if name == "ResNet18":
        return ResNet18(**model_config)
    raise ValueError(f"Unsupported model: {name}")


def build_criterion(config: dict[str, Any]) -> nn.Module:
    criterion_config = dict(config["training"]["criterion"])
    name = criterion_config.pop("name").lower()
    if name == "crossentropyloss":
        return nn.CrossEntropyLoss(**criterion_config)
    raise ValueError(f"Unsupported criterion: {name}")


def build_optimizer(config: dict[str, Any], model: nn.Module):
    optimizer_config = dict(config["training"]["optimizer"])
    name = optimizer_config.pop("name").lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **optimizer_config)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **optimizer_config)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(config: dict[str, Any], optimizer):
    scheduler_config = dict(config["training"]["scheduler"])
    name = scheduler_config.pop("name").lower()
    if name in {"none", "constant"}:
        return None
    if name == "cosineannealinglr":
        scheduler_config.setdefault("T_max", int(config["training"]["epochs"]))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **scheduler_config,
        )
    if name == "multisteplr":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
    raise ValueError(f"Unsupported scheduler: {name}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return None


def allocated_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024**2)
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024**2)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(run_dir.name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "console.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_metric(path: Path, row: dict[str, Any], create: bool) -> None:
    mode = "w" if create else "a"
    with path.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        if create:
            writer.writeheader()
        writer.writerow(row)


def load_previous_elapsed(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return 0.0
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return float(rows[-1]["elapsed_time_s"]) if rows else 0.0


def checkpoint_payload(
    config,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_epoch,
    best_val_acc,
):
    return {
        "run_id": config["run_id"],
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }


def prepare_new_run(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    run_id, run_dir = unique_run_dir(config)
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "figures").mkdir()
    config["run_id"] = run_id
    return config, run_dir


def run_experiment(
    config: dict[str, Any],
    run_dir: Path,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    device_name = config.get("runtime", {}).get("device", "auto")
    device = get_device(device_name)
    seed = int(config["seed"])
    deterministic = bool(config.get("runtime", {}).get("deterministic", False))
    set_seed(seed, deterministic)

    current_environment = environment_metadata(device)
    if resume_path and "environment" in config:
        config.setdefault("resume_history", []).append(current_environment)
    else:
        config["environment"] = current_environment
    atomic_write_json(run_dir / "config.json", config)
    logger = make_logger(run_dir)
    logger.info("Run: %s", config["run_id"])
    logger.info("Config: %s", run_dir / "config.json")
    logger.info("Device: %s", device)

    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {
        "run_id": config["run_id"],
        "experiment": config["experiment"],
        "variant": config["variant"],
        "display_name": config["display_name"],
        "seed": seed,
        "status": "running",
    }
    atomic_write_json(summary_path, summary)

    try:
        data_config = config["data"]
        data_dir = Path(data_config["data_dir"])
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir

        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=int(config["training"]["batch_size"]),
            val_ratio=float(data_config["val_ratio"]),
            seed=seed,
            num_workers=int(data_config["num_workers"]),
            augmentation=data_config["augmentation"],
            download=bool(data_config.get("download", True)),
            pin_memory=bool(data_config.get("pin_memory", device.type == "cuda")),
        )

        model = build_model(config).to(device)
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        criterion = build_criterion(config)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer)
        logger.info("Model: %s", config["model"])
        logger.info("Trainable parameters: %s (%.3f M)", num_parameters, num_parameters / 1e6)
        logger.info(
            "Samples: train=%s val=%s test=%s",
            len(train_loader.dataset),
            len(val_loader.dataset),
            len(test_loader.dataset),
        )

        metrics_path = run_dir / "metrics.csv"
        start_epoch = 1
        global_step = 0
        best_epoch = 0
        best_val_acc = float("-inf")
        previous_elapsed = 0.0

        if resume_path:
            state = torch_load(resume_path, device)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if scheduler and state.get("scheduler_state_dict"):
                scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch = int(state["epoch"]) + 1
            global_step = int(state["global_step"])
            best_epoch = int(state["best_epoch"])
            best_val_acc = float(state["best_val_acc"])
            previous_elapsed = load_previous_elapsed(metrics_path)
            logger.info("Resuming from epoch %s", start_epoch)

        total_start = time.perf_counter()
        epochs = int(config["training"]["epochs"])
        checkpoint_every = int(config["training"].get("checkpoint_every", 1))
        best_path = run_dir / "checkpoints" / "best_weights.pt"
        last_path = run_dir / "checkpoints" / "last_checkpoint.pt"

        if start_epoch > epochs:
            raise ValueError(
                f"Checkpoint is already at epoch {start_epoch - 1}, "
                f"but config requests only {epochs} epochs"
            )

        for epoch in range(start_epoch, epochs + 1):
            reset_peak_memory(device)
            synchronize(device)
            epoch_start = time.perf_counter()
            epoch_lr = float(optimizer.param_groups[0]["lr"])

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            synchronize(device)

            epoch_time = time.perf_counter() - epoch_start
            elapsed = previous_elapsed + (time.perf_counter() - total_start)
            global_step += len(train_loader)

            if val_acc > best_val_acc:
                best_epoch = epoch
                best_val_acc = val_acc
                atomic_torch_save(
                    {
                        "run_id": config["run_id"],
                        "config": config,
                        "epoch": epoch,
                        "best_val_acc": best_val_acc,
                        "model_state_dict": model.state_dict(),
                    },
                    best_path,
                )

            if scheduler:
                scheduler.step()

            row = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": epoch_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_s": epoch_time,
                "elapsed_time_s": elapsed,
                "allocated_memory_mb": allocated_memory_mb(device),
                "peak_memory_mb": peak_memory_mb(device),
            }
            write_metric(metrics_path, row, create=epoch == 1 and not resume_path)

            if epoch % checkpoint_every == 0 or epoch == epochs:
                atomic_torch_save(
                    checkpoint_payload(
                        config,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        best_epoch,
                        best_val_acc,
                    ),
                    last_path,
                )

            logger.info(
                "Epoch %03d/%03d | train %.4f / %.2f%% | "
                "val %.4f / %.2f%% | lr %.6f | %.1fs",
                epoch,
                epochs,
                train_loss,
                train_acc * 100,
                val_loss,
                val_acc * 100,
                epoch_lr,
                epoch_time,
            )

        total_time = previous_elapsed + (time.perf_counter() - total_start)
        test_loss = None
        test_acc = None
        if config["training"].get("evaluate_test", True):
            best_state = torch_load(best_path, device)
            model.load_state_dict(best_state["model_state_dict"])
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        summary.update(
            {
                "status": "completed",
                "model": config["model"]["name"],
                "device": str(device),
                "num_parameters": num_parameters,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "training_time_s": total_time,
                "best_weights": str(best_path.relative_to(run_dir)),
                "last_checkpoint": str(last_path.relative_to(run_dir)),
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        atomic_write_json(summary_path, summary)
        logger.info("Completed | best val %.2f%% at epoch %s", best_val_acc * 100, best_epoch)
        if test_acc is not None:
            logger.info("Test accuracy: %.2f%%", test_acc * 100)
        logger.info("Run directory: %s", run_dir)
        return summary
    except Exception as error:
        summary.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        atomic_write_json(summary_path, summary)
        logger.exception("Run failed")
        raise


def load_requested_config(args: argparse.Namespace):
    if args.resume:
        resume_path = args.resume.expanduser().resolve()
        run_dir = resume_path.parent.parent
        config = read_json(run_dir / "config.json")
        summary_path = run_dir / "summary.json"
        previous_summary = read_json(summary_path) if summary_path.exists() else {}
        if previous_summary.get("status") == "completed":
            raise ValueError("This run is already completed and should not be resumed")
        if args.seed is not None or args.epochs is not None:
            raise ValueError("--seed and --epochs cannot be changed while resuming")
        if args.device is not None:
            config.setdefault("runtime", {})["device"] = args.device
        if args.skip_test:
            config["training"]["evaluate_test"] = False
        return config, run_dir, resume_path

    config = deepcopy(read_json(args.config.expanduser().resolve()))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.device is not None:
        config.setdefault("runtime", {})["device"] = args.device
    if args.skip_test:
        config["training"]["evaluate_test"] = False
    validate_config(config)
    if args.dry_run:
        return config, None, None
    config, run_dir = prepare_new_run(config)
    return config, run_dir, None


def main() -> None:
    args = parse_args()
    config, run_dir, resume_path = load_requested_config(args)
    validate_config(config)

    if args.dry_run:
        model = build_model(config)
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(json.dumps(config, indent=2, ensure_ascii=False))
        print(f"\nConfig is valid. Trainable parameters: {num_parameters:,}")
        return

    run_experiment(config, run_dir, resume_path)


if __name__ == "__main__":
    main()
