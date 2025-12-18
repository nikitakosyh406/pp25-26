# code/json_to_mask.py
import os
import numpy as np
import cv2
import json

input_json_dir = r"C:\Users\ADMIN\OneDrive - УрФУ\Рабочий стол\учеба\pp25-26\road_powerline_segmentation\masks_manual"
output_mask_dir = r"C:\Users\ADMIN\mask_auto"

os.makedirs(output_mask_dir, exist_ok=True)

LABEL_TO_VALUE = {
    'road': 1,
    'powerline': 2,
}

print(f"📁 Обработка JSON из: {input_json_dir}")
print(f"💾 Сохранение масок в: {output_mask_dir}\n")

json_files = [f for f in os.listdir(input_json_dir) if f.lower().endswith('.json')]

if not json_files:
    print("❌ В папке masks_manual нет .json файлов!")
    exit()

for fname in json_files:
    json_path = os.path.join(input_json_dir, fname)
    mask_name = fname.replace('.json', '.png')
    output_path = os.path.join(output_mask_dir, mask_name)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Проверяем наличие обязательных полей
        if 'imageHeight' not in data or 'imageWidth' not in data:
            print(f"⚠️  В {fname} нет imageHeight/imageWidth — пропускаем")
            continue

        img_h = data['imageHeight']
        img_w = data['imageWidth']
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        for shape in data['shapes']:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.int32)
            value = LABEL_TO_VALUE.get(label, 0)
            if value == 0:
                continue
            cv2.fillPoly(mask, [points], value)

        # Проверяем, удалось ли сохранить
        success = cv2.imwrite(output_path, mask)
        if success:
            print(f"✅ {mask_name}")
        else:
            print(f"❌ Не удалось сохранить: {output_path} — проверь путь и права!")

    except Exception as e:
        print(f"💥 Ошибка при обработке {fname}: {e}")

print("\n✅ Готово!")

