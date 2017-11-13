#-*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import shutil

#水増しする画像のファイル名をtokyo.jpgのとこに入力

os.mkdir("picture_name")
shutil.copy("tokyo.jpg","picture_name")
os.chdir("./picture_name")
img = cv2.imread("tokyo.jpg",cv2.IMREAD_COLOR)

def main():
    
    mizumashi(5,5,3,3,5,5,3,3)

'''
    mizumashi(a,b,c,d,e,f,g,h)


各パラメータ

a = ガンマ変換(1<γ)　聖なる光に包まれます　可変パラメータはgamma1の0.1
b = ガンマ変換(1>γ)　ダークサイドへ堕ちます　可変パラメータはgamma2の0.05
                  γ>0でなくてはならないので、デフォルトでは(0<=b<20)

c = 平滑化縦方向　引き伸ばします　可変パラメータはaverage_squareの2
d = 平滑化横方向　平滑化のみ引数が2つになっており、c*d 枚出力します

e = アフィン変換(左回転)　左回りに回転します　可変パラメータはangle1の1
f = アフィン変換(右回転)　右回りに回転します　可変パラメータはangle2の1
g = 拡大　拡大します　可変パラメータはdstの0.05
h = 縮小　縮小します　可変パラメータはdstの0.05　(0<=h<=9)

出力枚数 = a+b+c*d+e+f+g+h (枚)
'''

def mizumashi(a,b,c,d,e,f,g,h):
    
    # ガンマ変換(1<γ)　聖なる光に包まれます　可変パラメータはgamma1の0.1
    for m in range(1,a+1):
        
        # ガンマ変換ルックアップテーブル
        LUT_G1 = np.arange(256, dtype = 'uint8' )
        gamma1 = 1+0.1*m
        for i in range(256):
            LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
            
        #　代入
        gamma1_img = cv2.LUT(img, LUT_G1)
        
        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(m), gamma1_img)
        
    # ガンマ変換(1>γ)　ダークサイドへ堕ちます　可変パラメータはgamma2の0.1
    for m in range(1,b+1):
        
        # ガンマ変換ルックアップテーブル
        LUT_G2 = np.arange(256, dtype = 'uint8' )
        gamma2 = 1-0.05*m
        for i in range(256):
            LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
            
        #　代入
        gamma2_img = cv2.LUT(img, LUT_G2)
        
        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+m), gamma2_img)
    
    #　平滑化 引き伸ばします　可変パラメータはaverage_squareの10
    #        平滑化のみ引数が2つになっており、e*f 枚出力します
    for m in range(1,c+1):
        for n in range(1,d+1):
            average_square = (2*m,2*n)
            
            #　代入
            blur_img = cv2.blur(img,average_square)
            
            # 結果を出力
            cv2.imwrite("tokyo{}.jpg".format(a+b+c*(m-1)+n), blur_img)
        
    #　アフィン変換(左回転)　左回りに回転します　可変パラメータはangle1の10など
    for m in range(1,e+1):
    
        # 画像のサイズ取得
        size = tuple([img.shape[1], img.shape[0]])

        # 回転の中心座標（画像の中心）
        center = tuple([int(size[0]/2), int(size[1]/2)])

        # 回転角度・拡大率
        angle1, scale = 1*m, 1.0 

        # 回転行列の計算
        R1 = cv2.getRotationMatrix2D(center, angle1, scale) 

        # アフィン変換
        dst1 = cv2.warpAffine(img, R1, size, flags=cv2.INTER_CUBIC) 

        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+c*d+m), dst1)
        
    #　アフィン変換(右回転)　右回りに回転します　可変パラメータはangle2の10など
    for m in range(1,f+1):
    
        # 画像のサイズ取得
        size = tuple([img.shape[1], img.shape[0]])

        # 回転の中心座標（画像の中心）
        center = tuple([int(size[0]/2), int(size[1]/2)])

        # 回転角度・拡大率
        angle2, scale = -1*m, 1.0  

        # 回転行列の計算
        R2 = cv2.getRotationMatrix2D(center, angle2, scale) 

        # アフィン変換
        dst2 = cv2.warpAffine(img, R2, size, flags=cv2.INTER_CUBIC) 

        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+c*d+e+m), dst2)
    
    #　拡大
    for m in range(1,g+1):
        dst3 = cv2.resize(img,None,fx=1+0.05*m, fy=1+0.05*m, interpolation = cv2.INTER_LINEAR)
        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+c*d+e+f+m), dst3)
    #　縮小
    for m in range(1,h+1):
        dst4 = cv2.resize(img,None,fx=1-0.05*m, fy=1-0.05*m, interpolation = cv2.INTER_AREA)
        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+c*d+e+f+g+m), dst4)


    
    
if __name__ == "__main__":
    main()