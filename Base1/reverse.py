#-*- coding:utf-8 -*-
import cv2
import numpy as np

def main():
    # 入力画像の読み込み
    img = cv2.imread("tokyo.jpg",cv2.IMREAD_COLOR)
    
    # 方法2(OpenCVで実装)       
    hflip_img = cv2.flip(img, 1)
    vflip_img = cv2.flip(img, 0)    
    # 結果を出力
    cv2.imwrite("tokyo1.jpg", hflip_img)
    cv2.imwrite("tokyo2.jpg", vflip_img)

if __name__ == "__main__":
    main()