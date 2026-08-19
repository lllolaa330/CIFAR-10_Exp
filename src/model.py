import torch.nn as nn

class CNNBaseline(nn.Module):
  def __init__(self, num_classes = 10):
    super().__init__()
    
    self.features = nn.Sequential(
      # 32 × 32
      nn.Conv2d(
        in_channels = 3,
        out_channels = 32,
        kernel_size = 3,
        padding = 1
      ),
      nn.ReLU(),
      
      # 32 × 32
      nn.Conv2d(
        in_channels = 32,
        out_channels = 64,
        kernel_size = 3,
        padding = 1
      ),
      nn.ReLU(),
      
      # 32 × 32 -> 16 × 16
      nn.MaxPool2d(kernel_size = 2),
      
      # 16 × 16
      nn.Conv2d(
        in_channels = 64,
        out_channels = 128,
        kernel_size = 3,
        padding = 1
      ),
      nn.ReLU(),
      
      # 16 × 16 -> 8 × 8
      nn.MaxPool2d(kernel_size = 2),
      
      # 8 × 8 -> 1 × 1
      nn.AdaptiveAvgPool2d(1),
    )
    
def forward(self, x):
  # input [B, 3, 32, 32]
  x = self.features(x)
  
  # [B, 128, 1, 1]
  x = x.flatten(1)
  
  # [B, 128]
  x = self.classifier(x)
  
  # [B, 10]
  return x
  