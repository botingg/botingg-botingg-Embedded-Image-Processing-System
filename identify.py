import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
import math

# 上傳圖片
files.upload()

img = cv2.imread('face4.webp')
if img is None:
    raise ValueError("圖片讀不到")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# Step 1: Gaussian Blur
# =========================
blur = cv2.GaussianBlur(gray, (9,9), 1.5)

# =========================
# Step 2: Hough Circle（抓瞳孔）
# =========================
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=100,
    param2=20,
    minRadius=5,
    maxRadius=50
)

# 複製圖畫結果
output = img.copy()

centers = []

if circles is not None:
    circles = np.uint16(np.around(circles))

    print("偵測到圓數量:", len(circles[0]))

    for (x, y, r) in circles[0]:
        # 畫圓
        cv2.circle(output, (x, y), r, (0,255,0), 2)
        # 畫中心
        cv2.circle(output, (x, y), 2, (0,0,255), 3)

        centers.append((x, y))

# =========================
# Step 3: 計算兩眼距離
# =========================
if len(centers) >= 2:
    (x1, y1) = centers[0]
    (x2, y2) = centers[1]

    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    print("左眼中心:", (x1, y1))
    print("右眼中心:", (x2, y2))
    print("兩眼距離:", distance)
else:
    print("偵測不到兩個瞳孔")

cv2_imshow(output)
