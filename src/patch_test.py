import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from timm.models.eva import PatchEmbed
from timm.models.tiny_vit import PatchEmbed as TinyPatchEmbed
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open('./image.png')
image = image.convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

image = transform(image)
image = image.unsqueeze(0)
print(image.shape)

patch_embed = PatchEmbed(256, 8, 3, 25, flatten=False)
output = patch_embed(image)

print(output.shape)

for i in range(output.shape[1]):
    plt.imshow(output[0, i].detach().numpy())
    plt.savefig(f'patch_{i}.png')

sum_output = output.sum(dim=1)
plt.imshow(sum_output[0].detach().numpy())
plt.savefig('sum_patch.png')