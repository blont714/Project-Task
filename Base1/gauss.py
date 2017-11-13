#-*- coding:utf-8 -*-
import cv2
import numpy as np

def main():
    # 入力画像の読み込み
    img = cv2.imread("tokyo.jpg",cv2.IMREAD_COLOR)
    
    # 方法2(OpenCVで実装)       
    row,col,ch= img.shape
    mean = 0
    sigma = 150
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss_img = img + gauss

    # 結果を出力
    cv2.imwrite("tokyo8.jpg", gauss_img)

if __name__ == "__main__":
    main()