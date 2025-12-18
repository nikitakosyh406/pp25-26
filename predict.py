# code/predict.py
import os
import torch
import torch.nn as nn
import cv2
import numpy as np

# ---------- КОПИЯ U-Net из train_unet.py ----------
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
    def __init__(self, n_classes=3):
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

# ---------- ОСНОВНОЙ КОД ----------
device = torch.device('cpu')

# Путь к модели (относительно predict.py)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'unet_road_powerline.pth')

# Загружаем модель
model = UNet(n_classes=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Пути к данным
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_dir = r"C:\Users\ADMIN\mask\images"
output_mask_dir = r"C:\Users\ADMIN\mask\masks_auto_full"
os.makedirs(output_mask_dir, exist_ok=True)

# Обработка изображений
for fname in os.listdir(image_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(image_dir, fname)
    out_path = os.path.join(output_mask_dir, os.path.splitext(fname)[0] + '.png')

    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Пропущено: не удалось загрузить {img_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    image_resized = cv2.resize(image, (256, 256)) / 255.0
    tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    pred_full = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, pred_full)
    print(f"✅ Создана маска: {out_path}")