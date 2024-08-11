import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.densenet import DenseBlock
from timm.models.vovnet import OsaBlock

class CombinedModel(nn.Module):
    def __init__(self, num_classes, in_chans=3, growth_rate=32, block_config=(6, 12, 24, 16)):
        super(CombinedModel, self).__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # OsaBlock
        self.osa_block = OsaBlock(in_chs=64, mid_chs=32, out_chs=128, layer_per_block=5)
        
        # DenseBlocks
        num_features = 128
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=4,
                growth_rate=growth_rate,
                drop_rate=0
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                self.features.add_module(f'transition{i + 1}', nn.Sequential(
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                ))
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.features[:4](x)  # Initial convolution and pooling
        osa_out = self.osa_block(features)
        dense_in = self.features[4:](osa_out)  # DenseBlocks and transitions
        out = F.relu(dense_in, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    model = CombinedModel(num_classes=5)
    x = torch.randn(8, 3, 224, 224)
    y = model(x)
    print(y.shape)
    print(f"Total parameters (M): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}")