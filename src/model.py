import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.densenet import densenet201, DenseNet
from timm.models.resnet import resnet152
from timm.models.vovnet import OsaBlock

class Model(nn.Module):
    def __init__(self, num_classes, in_chans=3):
        super(Model, self).__init__()
        self.background_denoiser = OsaBlock(in_chs=3, mid_chs=1, out_chs=3, layer_per_block=30)
        self.main_cnn:DenseNet = densenet201(num_classes=num_classes, in_chans=in_chans)
        # self.main_cnn = resnet152(num_classes=num_classes, in_chans=in_chans)

    def forward(self, x):
        noise = self.background_denoiser(x)
        noise = F.gelu(noise)
        x = x + noise
        return self.main_cnn(x)
    
if __name__ == "__main__":
    model = Model(num_classes=5)
    # model = OsaBlock(in_chs=3, mid_chs=9, out_chs=3, layer_per_block=20)
    x = torch.randn(8, 3, 256, 256)
    y = model(x)
    print(y.shape)
    print(f"Total parameters (M): {sum(p.numel() for p in model.parameters()) / 1e6}")