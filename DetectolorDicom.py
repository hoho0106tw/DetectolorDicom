

import os
import shutil
import pydicom
import numpy as np

SOURCE_DIR = r"\\10.80.64.22\ai2\Data\hoho\2ch"
TARGET_DIR = r"\\10.80.64.22\ai2\Data\hoho\2ch\color"

# 門檻設定（可調整）
MOTION_THRESHOLD = 25   # 移動偵測敏感度
COLOR_THRESHOLD = 30    # 色彩差異敏感度


def is_color_pixel(px):
    """偵測 px (RGB) 是否為紅 / 藍 / 黃"""
    r, g, b = px

    # 紅色
    if r > 150 and g < 100 and b < 100:
        return True

    # 藍色
    if b > 150 and r < 100 and g < 100:
        return True

    # 黃色
    if r > 150 and g > 150 and b < 80:
        return True

    return False


def detect_moving_color_area(pixel_data):
    """利用移動偵測 + 彩色偵測，在動態區塊中找紅/藍/黃"""

    # 必須是 multi-frame
    if pixel_data.ndim != 4:
        return False

    frame_count = pixel_data.shape[0]

    for i in range(frame_count - 1):
        f1 = pixel_data[i].astype(np.int16)
        f2 = pixel_data[i + 1].astype(np.int16)

        # 必須為 RGB
        if f1.ndim != 3 or f1.shape[-1] != 3:
            continue

        # ---------------------
        # ① 移動偵測
        # ---------------------
        diff = np.abs(f1 - f2)
        motion_mask = np.sum(diff, axis=2) > MOTION_THRESHOLD

        if not np.any(motion_mask):
            continue  # 這兩幀沒有移動

        # 擷取有移動的 pixel
        moving_pixels = f1[motion_mask]

        # ---------------------
        # ② 在移動區塊內找彩色（紅/藍/黃）
        # ---------------------
        for px in moving_pixels:
            if is_color_pixel(px):
                return True  # 找到動態彩色區塊

    return False


def main():
    for filename in os.listdir(SOURCE_DIR):
        if not filename.lower().endswith(".dcm"):
            continue

        filepath = os.path.join(SOURCE_DIR, filename)

        try:
            dcm = pydicom.dcmread(filepath)
            pixel_data = dcm.pixel_array
        except Exception as e:
            print(f"讀取錯誤 {filename}: {e}")
            continue

        try:
            if detect_moving_color_area(pixel_data):
                print(f"→ 偵測到動態彩色區塊，移動：{filename}")
                shutil.move(filepath, os.path.join(TARGET_DIR, filename))
            else:
                print(f"{filename} 沒有動態彩色區塊")
        except Exception as e:
            print(f"處理錯誤 {filename}: {e}")


if __name__ == "__main__":
    main()
