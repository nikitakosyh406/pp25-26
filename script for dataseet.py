import os
import math
import requests
from PIL import Image
from io import BytesIO

# === Настройки ===
CITY_NAME = "moscow"
OUTPUT_DIR = f"dataset/{CITY_NAME}"
ZOOM = 18  # Повысим детализацию: ~0.6 м/пикс → тайл ≈ 150×150 м
MAX_TILES = 200  # Теперь 200 тайлов

# Центр Москвы (Красная площадь)
CENTER_LAT = 55.7558
CENTER_LON = 37.6176

# Радиус в тайлах (подберём так, чтобы хватило на 200+)
# Например, 9×9 = 81, 15×15 = 225 → возьмём RADIUS=12 → 25×25=625 (но ограничим 200)
RADIUS_TILES = 12

# URL Esri (без токена)
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def deg2num(lat_deg, lon_deg, zoom):
    """Переводит (широта, долгота) в номер тайла XYZ"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x, y

def download_tile(x, y, z):
    url = TILE_URL.format(z=z, y=y, x=x)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SatelliteBot/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f"Ошибка {response.status_code} при загрузке {x},{y}")
            return None
    except Exception as e:
        print(f"Исключение при загрузке {x},{y}: {e}")
        return None

# === Основная логика ===
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    center_x, center_y = deg2num(CENTER_LAT, CENTER_LON, ZOOM)
    print(f"Центр Москвы → тайл: x={center_x}, y={center_y} (zoom={ZOOM})")
    print(f"Будет скачано до {MAX_TILES} тайлов...")

    tile_count = 0
    for dy in range(-RADIUS_TILES, RADIUS_TILES + 1):
        for dx in range(-RADIUS_TILES, RADIUS_TILES + 1):
            if tile_count >= MAX_TILES:
                print(f"✅ Достигнут лимит: {MAX_TILES} тайлов.")
                break

            x = center_x + dx
            y = center_y + dy

            img = download_tile(x, y, ZOOM)
            if img:
                filename = f"{CITY_NAME}_z{ZOOM}_x{x}_y{y}.png"
                img.save(os.path.join(OUTPUT_DIR, filename))
                tile_count += 1
                if tile_count % 20 == 0:
                    print(f"📥 Скачано: {tile_count} / {MAX_TILES}")
        else:
            continue
        break

    print(f"🎉 Готово! {tile_count} тайлов сохранено в: {OUTPUT_DIR}")