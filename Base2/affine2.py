#-*- coding:utf-8 -*-
import cv2
import numpy as np

def main():
    # 入力画像の読み込み
    img = cv2.imread("tokyo.jpg",cv2.IMREAD_COLOR)
    
    for m in range(1,6):
    
        # 画像のサイズ取得
        size = tuple([img.shape[1], img.shape[0]])

        # 回転の中心座標（画像の中心）
        center = tuple([int(size[0]/2), int(size[1]/2)])

        # 回転角度・拡大率
        angle1, scale = 10*m, 1.0
        angle2, scale = -10*m, 1.0  

        # 回転行列の計算
        R1 = cv2.getRotationMatrix2D(center, angle1, scale) 
        R2 = cv2.getRotationMatrix2D(center, angle2, scale) 

        # アフィン変換
        dst1 = cv2.warpAffine(img, R1, size, flags=cv2.INTER_CUBIC) 
        dst2 = cv2.warpAffine(img, R2, size, flags=cv2.INTER_CUBIC) 

        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(2*m-1), dst1)
        cv2.imwrite("tokyo{}.jpg".format(2*m), dst2)


if __name__ == "__main__":
    main()