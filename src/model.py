import torch
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
    
    self.classifier = nn.Linear(
      in_features = 128,
      out_features=num_classes
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

class BasicBlock(nn.Module):
  expansion = 1
  def __init__(self, in_channels, out_channels, stride =1):
    super().__init__()
    self.conv1 = nn.Conv2d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=3,
      stride=stride,
      padding=1,
      bias=False,
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(
      in_channels=out_channels,
      out_channels=out_channels,
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False,
    )
    self.bn2 = nn.BatchNorm2d(out_channels)
    
    # Shortcut Branch
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=1,
          stride=stride,
          bias=False,
        ),
        nn.BatchNorm2d(out_channels),
      )
    else:
      self.shortcut = nn.Identity()
    
  def forward(self, x):
    identity = self.shortcut(x)
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    
    out = out + identity
    
    out = self.relu(out)
    return out
    
class ResNet18(nn.Module):
  def __init__(self, num_classes = 10):
    super().__init__()
    
    # CIFAR-10:
    # input = [B, 3, 32, 32]
    # 不使用 ImageNet ResNet  7×7 Conv + stride=2 + MaxPool
    # 而使用 CIFAR-style:     3×3 Conv + stride=1
    self.in_channels = 64
    self.conv1 = nn.Conv2d(
      in_channels=3,
      out_channels=64,
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False,
    )
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    # Residual Layers
    self.layer1 = self._make_layer(
      out_channels = 64,
      blocks = 2,
      stride = 1,
    )
    self.layer2 = self._make_layer(
      out_channels = 128,
      blocks = 2,
      stride = 2,
    )
    self.layer3 = self._make_layer(
      out_channels = 256,
      blocks = 2,
      stride = 2,
    )
    self.layer4 = self._make_layer(
      out_channels = 512,
      blocks = 2,
      stride = 2,
    )
    
    # Classification Head
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(
      in_features = 512 * BasicBlock.expansion,
      out_features = num_classes,
    )
    
    # Weight Initialization
    self._initialize_weights()
  
  def _make_layer(self, out_channels, blocks, stride):
    layers = []
    layers.append(
      BasicBlock(
        in_channels=self.in_channels, 
        out_channels=out_channels,
        stride=stride,
      )
    )
    self.in_channels=(out_channels * BasicBlock.expansion)
    for _ in range(1, blocks):
      layers.append(
        BasicBlock(
          in_channels=self.in_channels,
          out_channels=out_channels,
          stride=1,
        )
      )
    return nn.Sequential(*layers)
  
  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
          m.weight,
          mode = "fan_out",
          nonlinearity = "relu",
        )
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.normal_(m.bias, 0)
    
  def forward(self, x):
    # input [B, 3, 32, 32]
    
    # [B, 64, 32, 32]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    
    # Residual Layers
    # [B, 64, 32, 32]
    x = self.layer1(x)
    # [B, 128, 16, 16]
    x = self.layer2(x)
    # [B, 256, 8, 8]
    x = self.layer3(x)
    # [B, 512, 4, 4]
    x = self.layer4(x)
    
    # Global Average Pooling
    # [B, 512, 1, 1]
    x = self.avgpool(x)
    # [B, 512]
    x = torch.flatten(x, 1)
    # [B, 10]
    x = self.fc(x)
    
    return x
  