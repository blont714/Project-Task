#-*- coding:utf-8 -*-
import cv2
import numpy as np

# ガンマ変換ルックアップテーブル
LUT_G1 = np.arange(256, dtype = 'uint8' )
LUT_G2 = np.arange(256, dtype = 'uint8' )
gamma1 = 0.50
gamma2 = 2
for i in range(256):
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

def main():
    # 入力画像の読み込み
    img = cv2.imread("tokyo.jpg",cv2.IMREAD_COLOR)
    
    # 方法2(OpenCVで実装)       
    gamma1_img = cv2.LUT(img, LUT_G1)
    gamma2_img = cv2.LUT(img, LUT_G2)
    # 結果を出力
    cv2.imwrite("tokyo5.jpg", gamma1_img)
    cv2.imwrite("tokyo6.jpg", gamma2_img)

if __name__ == "__main__":
    main()