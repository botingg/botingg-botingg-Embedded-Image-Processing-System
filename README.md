# 作業：人臉瞳孔偵測
## 開發環境
https://code.visualstudio.com/

## 程式碼
```python=
import cv2
import numpy as np
import math

# =========================
# Step 0: 讀取圖片（本地路徑）
# =========================
img = cv2.imread('face4.webp')  # 確認圖片和程式在同一資料夾
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

    # 避免 overflow，轉 int
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    print("左眼中心:", (x1, y1))
    print("右眼中心:", (x2, y2))
    print("兩眼距離:", distance)
else:
    print("偵測不到兩個瞳孔")

# =========================
# Step 4: 顯示結果
# =========================
cv2.imshow("Pupil Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## 兩眼距離公式
```py
distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
```

## Tools流程
+ **高斯模糊（Gaussian Blur）**
+ **霍夫圓轉換（Hough Circle Transform）**


![image](identify_image/face.png)

**Saving face.png to face (1).png**
**偵測到圓數量: 2**

**左眼中心: (np.uint16(106), np.uint16(280))**

**右眼中心: (np.uint16(284), np.uint16(236))**

**兩眼距離: 183.35757415498276**

![image](identify_image/face1.png)

**Saving face1.png to face1.png**
**偵測到圓數量: 2**

**左眼中心: (np.uint16(424), np.uint16(126))**

**右眼中心: (np.uint16(222), np.uint16(120))**

**兩眼距離: 202.08908926510605**

![image](identify_image/face2.png)

**Saving face2.png to face2.png**

**偵測到圓數量: 2**

**左眼中心: (np.uint16(224), np.uint16(68))**

**右眼中心: (np.uint16(94), np.uint16(50))**

**兩眼距離: 131.24023773218335**

![image](identify_image/face3.png)

**Saving face3.jpg to face3 (1).jpg**

**偵測到圓數量: 2**

**左眼中心: (np.uint16(282), np.uint16(240))**

**右眼中心: (np.uint16(384), np.uint16(242))**

**兩眼距離: 102.0196059588548**


![image](identify_image/face4.png)


**左眼: (203, 221)**

**右眼: (333, 255)**

**瞳孔距離: 134.3726162579266**





## 為啥只用兩種Tools？
+ **霍夫圓演算法本身就是針對「圓形」特徵設計的，能直接偵測圓的中心與半徑。**
+ **高斯模糊只是減少雜訊，讓霍夫圓偵測時不會被小的細節或噪點干擾。**
+ **簡單的處理，保留了完整圓形輪廓訊號 → 圓形檢測精準。**

### **(a) Sobel/Canny 對邊緣敏感**
+ Sobel 偵測梯度，Canny 找出邊緣。
+ 瞳孔邊界通常對比不高、光照有反射，會導致：
+ 邊緣斷裂
+ 邊緣過多（噪點）或過少（弱邊緣消失）
+ 霍夫圓需要完整的圓形邊緣訊號，邊緣不連續就檢測不到。

### **(b) 四角（Perspective Transform / 仿射轉換）**
+ 通常用於校正圖像角度。
+ 若角度校正不精準，會破壞圓形比例 → 圓變橢圓或變形。
+ 霍夫圓對圓形變形非常敏感 → 偵測失敗。

### **(c) 太多處理步驟 → 累積誤差**
+ 高斯模糊 → Sobel → Canny → Threshold → Contour → Hough
+ 每個步驟都會丟掉一些像素資訊
+ 結果：瞳孔邊界被破壞 → 霍夫圓找不到真正的圓心


