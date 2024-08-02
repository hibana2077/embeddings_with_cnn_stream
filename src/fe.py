import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from model import Model
from timm.models.convnext import convnext_atto
from timm.models.densenet import densenet201
from timm.models.efficientnet import efficientnet_b0
from timm.models.resnet import resnet18, resnet152
import os
import json

# define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# check './data' directory
if not os.path.exists('./data'):
    os.makedirs('./data')

# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define dataset
class MangoDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = ['A', 'B', 'C']
        self.split_mapping = {
            "train": "C1-P1_Train",
            "test": "C1-P1_Test",
            "dev": "C1-P1_Dev"
        }
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # read data
        self._read_data()

    def _read_data(self):
        # data
        # - C1-P1_Train
        #   - Images...
        # - Label data
        #   - C1-P1_Train.csv

        # read label data
        label_data = pd.read_csv(os.path.join(self.root, 'Label data', f'{self.split_mapping[self.split]}.csv'))
        for idx, row in label_data.iterrows():
            img_path = os.path.join(self.root, self.split_mapping[self.split], row['image_id'])
            label = self.class_to_idx[row['label']]
            self.data.append(img_path)
            self.labels.append(label)

    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

train_dataset = MangoDataset(root='../data', split='train', transform=transform)
test_dataset = MangoDataset(root='../data', split='dev', transform=transform)

# define dataloader
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = torch.load('model.pth')

import matplotlib.pyplot as plt

for i, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)
    output = model(img)
    background_denoiser_output = model.background_denoiser(img)
    background_denoiser_output = F.gelu(background_denoiser_output)
    _, predicted = output.max(1)
    print(f"Predicted: {predicted}")
    print(f"Ground Truth: {label}")
    # make img for visualization
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (0, 2, 3, 1))
    img = img[0]
    img = np.clip(img, 0, 1)
    background_denoiser_output = background_denoiser_output.cpu().detach().numpy()
    background_denoiser_output = np.transpose(background_denoiser_output, (0, 2, 3, 1))
    background_denoiser_output = background_denoiser_output[0]
    background_denoiser_output = np.clip(background_denoiser_output, 0, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted}, Ground Truth: {label}")
    plt.savefig(f"img_{i}.png")
    plt.imshow(background_denoiser_output)
    plt.title("Background Denoiser Output")
    plt.savefig(f"background_denoiser_output_{i}.png")
    if i == 10:
        break