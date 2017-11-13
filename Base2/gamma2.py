#-*- coding:utf-8 -*-
import cv2
import numpy as np

def main():
    # 入力画像の読み込み
    img = cv2.imread("tokyo.jpg",cv2.IMREAD_COLOR)
    
    
    # ガンマ変換ルックアップテーブル
    for m in range(1,10):
        LUT_G1 = np.arange(256, dtype = 'uint8' )
        LUT_G2 = np.arange(256, dtype = 'uint8' )
        gamma1 = 1-0.1*m
        gamma2 = 1+0.1*m
        for i in range(256):
            LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
            LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
    # 方法2(OpenCVで実装)       
        gamma1_img = cv2.LUT(img, LUT_G1)
        gamma2_img = cv2.LUT(img, LUT_G2)
    # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(2*m-1), gamma1_img)
        cv2.imwrite("tokyo{}.jpg".format(2*m), gamma2_img)

if __name__ == "__main__":
    main()