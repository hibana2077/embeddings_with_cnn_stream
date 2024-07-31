import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
from timm.models.densenet import densenet201, DenseNet

class Model(nn.Module):
    def __init__(self, num_classes, img_size=256, patch_size=32, in_chans=3):
        super(Model, self).__init__()
        self.main_cnn:DenseNet = densenet201(num_classes=num_classes, in_chans=in_chans)
        self.patchembed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=64
        )

    def forward(self, x):
        embed_tensor = self.patchembed(x) # (B, 64, 64)
        embed_tensor = embed_tensor.reshape(embed_tensor.shape[0], embed_tensor.shape[1], 8, 8)
        image_features = self.main_cnn.forward_features(x) # (B, 1920, 8, 8)
        fusion = torch.cat((embed_tensor, image_features), dim=1)
        return fusion
    
if __name__ == "__main__":
    model = Model(num_classes=5)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)