#-*- coding:utf-8 -*-
import cv2
import numpy as np

# ルックアップテーブルの生成
min_table = 50
max_table = 205
diff_table = max_table - min_table

LUT_HC = np.arange(256, dtype = 'uint8' )
LUT_LC = np.arange(256, dtype = 'uint8' )

# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    LUT_HC[i] = 255
                
# ローコントラストLUT作成
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255

def main():
# 変換
    img = cv2.imread("tokyo.jpg",cv2.IMREAD_COLOR)
    high_cont_img = cv2.LUT(img, LUT_HC)
    low_cont_img = cv2.LUT(img, LUT_LC)
    cv2.imwrite("tokyo3.jpg", high_cont_img)
    cv2.imwrite("tokyo4.jpg", low_cont_img)
    
if __name__ == "__main__":
    main()
