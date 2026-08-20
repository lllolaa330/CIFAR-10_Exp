# Experiment workflow

Run every command below from the project root. One training command creates one
immutable directory under `runs/`; it never overwrites a previous experiment.

## Output of one run

```text
runs/<run_id>/
├── config.json
├── console.log
├── metrics.csv
├── summary.json
├── checkpoints/
│   ├── best_weights.pt
│   └── last_checkpoint.pt
└── figures/
```

- `config.json` is the resolved, frozen experiment recipe.
- `metrics.csv` is the source data for figures.
- `summary.json` contains the best epoch, final test result, parameter count,
  runtime, and completion status.
- `best_weights.pt` is the compact evaluation checkpoint.
- `last_checkpoint.pt` includes optimizer and scheduler state for resuming.
- Figures are derived artifacts and can always be regenerated.

## Validate a config without training

```bash
python src/train.py --config configs/baseline.json --dry-run
python src/train.py --config configs/resnet18.json --dry-run
```

## Rebuild the two reference models

```bash
python src/train.py --config configs/baseline.json
python src/train.py --config configs/resnet18.json
```

The rebuilt ResNet-18 is the new corrected reference: its final linear bias is
initialized to zero. Legacy ResNet results used a unit-standard-deviation random
bias due to the old initialization call, so old and rebuilt results should not
be mixed in one formal ablation table.

Plot each completed run by substituting its actual directory:

```bash
python src/plot.py list
python src/plot.py run runs/<run_id>
```

Compare the reference models:

```bash
python src/plot.py compare \
  --runs runs \
  --experiment model_comparison \
  --metric val_acc \
  --output reports/model_comparison
```

## ResNet-18 2x2x2 ablation

All eight runs use the fixed seed `266978`. The three factors are classifier
dropout (`p=0.3`), random crop plus horizontal flip, and reducing the base
channels from 64 to 32. Dropout is applied after global average pooling and
before the final linear classifier.

| Config | Dropout | Crop + flip | Base channels |
| --- | :---: | :---: | ---: |
| `resnet18_reference.json` | no | no | 64 |
| `resnet18_dropout.json` | yes | no | 64 |
| `resnet18_crop_flip.json` | no | yes | 64 |
| `resnet18_width32.json` | no | no | 32 |
| `resnet18_dropout_crop_flip.json` | yes | yes | 64 |
| `resnet18_dropout_width32.json` | yes | no | 32 |
| `resnet18_width32_crop_flip.json` | no | yes | 32 |
| `resnet18_dropout_crop_flip_width32.json` | yes | yes | 32 |

Run the eight experiments in this order. Each run selects its checkpoint using
validation accuracy and evaluates the test set once at the end:

```bash
python src/train.py --config configs/ablations/resnet18_reference.json
python src/train.py --config configs/ablations/resnet18_dropout.json
python src/train.py --config configs/ablations/resnet18_crop_flip.json
python src/train.py --config configs/ablations/resnet18_width32.json
python src/train.py --config configs/ablations/resnet18_dropout_crop_flip.json
python src/train.py --config configs/ablations/resnet18_dropout_width32.json
python src/train.py --config configs/ablations/resnet18_width32_crop_flip.json
python src/train.py --config configs/ablations/resnet18_dropout_crop_flip_width32.json
```

Compare all eight runs:

```bash
python src/plot.py compare \
  --runs runs \
  --experiment resnet18_ablation \
  --metric val_acc \
  --output reports/resnet18_ablation
```

Repeat with `--metric val_loss`, `train_acc`, `epoch_time_s`,
`allocated_memory_mb`, or `peak_memory_mb`. CUDA records both allocated and
peak memory; MPS records allocated memory because PyTorch does not expose the
same peak-memory counter there. The comparison command also writes
`comparison.csv` for tables and further analysis.

The eight configurations are fixed before training. Use validation accuracy for
model selection and treat each run's single final test result as its reportable
generalization result; do not create further variants based on test accuracy.

## Resume an interrupted run

```bash
python src/train.py \
  --resume runs/<run_id>/checkpoints/last_checkpoint.pt
```

## Legacy artifacts

Old logs, checkpoints, and figures are preserved under `archive/legacy/`.
`archive/legacy/manifest.csv` records what can and cannot be inferred. Items
marked `inferred` or `orphaned` should not be used as fully reproducible evidence.
