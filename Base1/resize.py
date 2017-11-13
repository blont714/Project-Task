# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os

def main():
    # 入力画像の読み込み
    img = cv2.imread("input.jpg")

    # 方法2(OpenCV)
    res = cv2.resize(img,None,fx=0.75, fy=0.75, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("input1.jpg",res)

    files = os.listdir('C:\Users\BLOND\.spyder2')
    print (files)

if __name__ == "__main__":
    main()