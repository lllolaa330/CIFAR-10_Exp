import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# CIFAR-10 官方常用的 normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_dataloaders(
  data_dir = "/Users/jiaojiayi/Desktop/2608.week3/cifar10-resnet/data/cifar-10-batches-py",
  batch_size = 128,
  val_ratio = 0.1,
  seed = 266978,
  num_workers = 0,
):
  # Transform
  transform = transforms.Compose([
    # 改变维度排列 & 数值归一化
    transforms.ToTensor(),
    # 逐通道的标准化（Z-score 标准化）
    transforms.Normalize(
      mean = CIFAR10_MEAN,
      std = CIFAR10_STD
    ) 
  ])
  
  # Load CIFAR-10
  full_train_dataset = datasets.CIFAR10(
    root = data_dir,
    train = True,
    download = True,
    transform = transform,
  )
  
  test_dataset = datasets.CIFAR10(
    root = data_dir,
    train = False,
    download = True,
    transform = transform,
  )
  
  # Data Split
  num_train = len(full_train_dataset)
  num_val = int(num_train * val_ratio)
  num_train = num_train - num_val
  
  generator = torch.Generator().manual_seed(seed)
  
  train_dataset, val_dataset = torch.utils.data.random_split(
    full_train_dataset,
    [num_train, num_val],
    generator = generator,
  )
  
  # DataLoader
  train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
  )
  
  val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers,
  )
  
  test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers,
  )
  
  return train_loader, val_loader, test_loader