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
from timm.models.efficientnet import efficientnet_b0
from timm.models.resnet import resnet18
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
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

# check dataset size
print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))

for i, (img, label) in enumerate(train_loader):
    print(f"Batch {i} - Image shape: {img.shape}, Label shape: {label.shape}")
    break

# define model
model = Model(num_classes=3)
# model = resnet18(num_classes=3)
model = model.to(device)

# define loss function
criterion_cls = nn.CrossEntropyLoss()

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
LR_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# define training function
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs_cls = model(inputs)
        loss_cls = criterion_cls(outputs_cls, targets)
        loss = loss_cls
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs_cls.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print(f"[Train] Epoch: {epoch}, Batch: {batch_idx}, Loss: {train_loss/total:.4f}, Accuracy: {100. * correct / total:.4f}",end='\r')
    
    print(f"[Train] Epoch: {epoch}, Batch: {batch_idx}, Loss: {train_loss/total:.4f}, Accuracy: {100. * correct / total:.4f}")
    return train_loss/total, 100. * correct / total

# define testing function
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_cls = model(inputs)
            loss_cls = criterion_cls(outputs_cls, targets)
            loss = loss_cls
            test_loss += loss.item()
            _, predicted = outputs_cls.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {test_loss/total:.4f}, Accuracy: {100. * correct / total:.4f}",end='\r')

    print(f"[Test] Epoch: {epoch}, Batch: {batch_idx}, Loss: {test_loss/total:.4f}, Accuracy: {100. * correct / total:.4f}")
    return test_loss/total, 100. * correct/total

# define main function
def main():
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    for epoch in range(5):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        LR_scheduler.step()
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

    # save training history
    history = {
        'train_loss': train_loss_hist,
        'train_acc': train_acc_hist,
        'test_loss': test_loss_hist,
        'test_acc': test_acc_hist
    }
    with open('history.json', 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    main()