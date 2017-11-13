#-*- coding:utf-8 -*-
import cv2
import numpy as np

def main():
    # 入力画像の読み込み
    img = cv2.imread("tokyo.jpg",cv2.IMREAD_COLOR)
    
    # 方法2(OpenCVで実装)
    #(縦方向のブレ,横方向のブレ)
    for m in range(1,6):
        for n in range(1,6):
            average_square = (10*m,10*n)
            blur_img = cv2.blur(img,average_square)
    # 結果を出力
            cv2.imwrite("tokyo{}.jpg".format(5*(m-1)+n), blur_img)

if __name__ == "__main__":
    main()