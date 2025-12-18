# code/roads_enhanced_visualize.py
import os
import cv2
import numpy as np

# Пути
image_dir = r"C:\Users\ADMIN\mask\images"
mask_dir = r"C:\Users\ADMIN\mask\masks_auto_full"
vis_dir = r"C:\Users\ADMIN\mask\roads_enhanced_vis"
os.makedirs(vis_dir, exist_ok=True)

for fname in os.listdir(image_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + '.png')

    if not os.path.exists(mask_path):
        print(f"⚠️ Пропущено: маска не найдена — {mask_path}")
        continue

    # Загружаем изображение и маску
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"⚠️ Ошибка загрузки: {fname}")
        continue

    # === ШАГ 1: Осветление изображения (улучшаем видимость) ===
    # Метод: коррекция по каналам + повышение яркости
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Увеличиваем значение (Value) — яркость
    v = cv2.add(v, 30)  # +30 — можно регулировать
    v = np.clip(v, 0, 255)

    # Увеличиваем насыщенность (Saturation), если нужно
    s = cv2.add(s, 10)
    s = np.clip(s, 0, 255)

    enhanced_hsv = cv2.merge([h, s, v])
    bright_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # === ШАГ 2: Наложение дорог (ярко-красные контуры) ===
    result = bright_img.copy()

    # Выделяем только дороги (класс 1)
    road_mask = (mask == 1).astype(np.uint8) * 255

    # Находим контуры
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем **толстые красные контуры**
    cv2.drawContours(result, contours, -1, (0, 0, 255), thickness=3)  # BGR: красный

    # === ШАГ 3: Сохранение результата ===
    out_path = os.path.join(vis_dir, fname)
    cv2.imwrite(out_path, result)
    print(f"✅ Сохранено: {out_path}")