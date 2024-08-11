import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.eva import eva02_tiny_patch14_224, EvaBlock


if __name__ == "__main__":
    model = Model(num_classes=5)
    # model = OsaBlock(in_chs=3, mid_chs=9, out_chs=3, layer_per_block=20)
    x = torch.randn(8, 3, 256, 256)
    y = model(x)
    print(y.shape)
    print(f"Total parameters (M): {sum(p.numel() for p in model.parameters()) / 1e6}")
    print(eva02_tiny_patch14_224().blocks[0](torch.randn(8, 192, 192)).shape)
    print(eva02_tiny_patch14_224().blocks[0])