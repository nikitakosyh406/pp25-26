# code/train_unet.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from albumentations import *
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ---------- U-Net модель ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=3):  # 0=фон, 1=дорога, 2=ЛЭП → 3 класса
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.upconv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(nn.MaxPool2d(2)(x1))
        x3 = self.down3(nn.MaxPool2d(2)(x2))
        x4 = self.down4(nn.MaxPool2d(2)(x3))
        x5 = self.bottleneck(nn.MaxPool2d(2)(x4))

        u1 = self.up1(x5)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.upconv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.upconv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.upconv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.upconv4(u4)

        return self.outc(u4)

# ---------- Датасет ----------
class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.names = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(name)[0] + '.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()

# ---------- Аугментации ----------
train_transform = Compose([
    Resize(256, 256),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Rotate(limit=15, p=0.3),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ---------- Обучение ----------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # 🔴 УКАЖИ СВОИ ПУТИ ЗДЕСЬ 🔴
    image_dir = r"C:\Users\ADMIN\mask\images1"
    mask_dir = r"C:\Users\ADMIN\mask\mask_auto"

    dataset = SegDataset(image_dir=image_dir, mask_dir=mask_dir, transform=train_transform)
    if len(dataset) == 0:
        raise ValueError("❌ Нет изображений или масок! Проверь пути.")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = UNet(n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"📊 Найдено изображений: {len(dataset)}")
    print("🚀 Начинаю обучение...")

    model.train()
    for epoch in range(20):  # Обучение: 20 эпох
        epoch_loss = 0
        for images, masks in tqdm(dataloader, desc=f"Эпоха {epoch+1}/20"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Средний loss: {avg_loss:.4f}")

    # Сохраняем модель
    os.makedirs(r"C:\Users\ADMIN\OneDrive - УрФУ\Рабочий стол\учеба\pp25-26\road_powerline_segmentation\models", exist_ok=True)
    model_path = r"C:\Users\ADMIN\OneDrive - УрФУ\Рабочий стол\учеба\pp25-26\road_powerline_segmentation\models\unet_road_powerline.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Модель сохранена: {model_path}")

if __name__ == "__main__":
    train()