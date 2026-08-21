from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


torch.manual_seed(266978)

dataset = datasets.CIFAR10(
    root="data/cifar-10-batches-py",
    train=True,
    download=False,
)

image, label = dataset[1]

# 与实验配置一致：padding=4 后随机裁剪回 32×32
cropped = transforms.RandomCrop(32, padding=4)(image)

# 为了展示效果，这里强制翻转
# 实际训练中的 RandomHorizontalFlip 默认以 0.5 概率翻转
flipped = TF.hflip(image)

images = [image, cropped, flipped]
titles = [
    f"Original: {dataset.classes[label]}",
    "Random Crop(padding = 4)",
    "Horizontal Flip",
]

fig, axes = plt.subplots(1, 3, figsize=(8, 3))

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

fig.tight_layout()

output = Path("reports/data_augmentation_example.png")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved to {output}")