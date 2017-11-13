#-*- coding:utf-8 -*-
import cv2
import numpy as np

def main():
    # 入力画像の読み込み
    img = cv2.imread("tokyo.jpg",cv2.IMREAD_COLOR)
# 変換
    for m in range(1,11):
    # ルックアップテーブルの生成
        min_table = m
        max_table = 255 - m
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
        
        high_cont_img = cv2.LUT(img, LUT_HC)
        low_cont_img = cv2.LUT(img, LUT_LC)
        cv2.imwrite("tokyo{}.jpg".format(2*m-1), high_cont_img)
        cv2.imwrite("tokyo{}.jpg".format(2*m), low_cont_img)
    
if __name__ == "__main__":
    main()
