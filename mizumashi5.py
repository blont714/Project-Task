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
    
    mizumashi(1,1,1,1,1,1,1,1,1,1,1)

'''
    mizumashi(a,b,c,d,e,f,g,h,j,k,l)


各パラメータ
a = コントラスト調整(強調)　いかつい画像になります　可変パラメータはmin_tableとmax_tableの10
b = コントラスト調整(低減)　PM2.5が舞い踊ります　可変パラメータはmin_tableとmax_tableの10
c = ガンマ変換(1<γ)　聖なる光に包まれます　可変パラメータはgamma1の0.1
d = ガンマ変換(1>γ)　ダークサイドへ堕ちます　可変パラメータはgamma2の0.1
                  γ>0でなくてはならないので、デフォルトでは(0<=d<=9)

e = 平滑化縦方向　引き伸ばします　可変パラメータはaverage_squareの10
f = 平滑化横方向　平滑化のみ引数が2つになっており、e*f 枚出力します

g = ガウスノイズ付加 粗くなります　可変パラメータはsigmaの15
h = アフィン変換(左回転)　左回りに回転します　可変パラメータはangle1の10
j = アフィン変換(右回転)　右回りに回転します　可変パラメータはangle2の10
k = 拡大　拡大します　可変パラメータはdstの0.1
l = 縮小　縮小します　可変パラメータはdstの0.1　(0<=l<=9)

出力枚数 = a+b+c+d+e*f+g+h+j (枚)
'''

def mizumashi(a,b,c,d,e,f,g,h,j,k,l):
    
    #　コントラスト調整(強調)　いかつい画像になります　可変パラメータはmin_tableとmax_tableの10
    for m in range(1,a+1):
        
        # ルックアップテーブルの生成
        min_table1 = 10*m
        max_table1 = 255 - 10*m
        diff_table1 = max_table1 - min_table1

        LUT_HC = np.arange(256, dtype = 'uint8' )

        # ハイコントラストLUT作成
        for i in range(0, min_table1):
            LUT_HC[i] = 0
        for i in range(min_table1, max_table1):
            LUT_HC[i] = 255 * (i - min_table1) / diff_table1
        for i in range(max_table1, 255):
            LUT_HC[i] = 255
            
        #　代入
        high_cont_img = cv2.LUT(img, LUT_HC)
        
        # 結果出力
        cv2.imwrite("tokyo{}.jpg".format(m), high_cont_img)
        
    #　コントラスト調整(低減)　PM2.5が舞い踊ります　可変パラメータはmin_tableとmax_tableの10
    for m in range(1,b+1):
        
        # ルックアップテーブルの生成
        min_table2 = 10*m
        max_table2 = 255 - 10*m
        diff_table2 = max_table2 - min_table2

        LUT_LC = np.arange(256, dtype = 'uint8' )
                
        # ローコントラストLUT作成
        for i in range(256):
            LUT_LC[i] = min_table2 + i * (diff_table2) / 255
            
        #　代入
        low_cont_img = cv2.LUT(img, LUT_LC)
        
        # 結果出力
        cv2.imwrite("tokyo{}.jpg".format(a+m), low_cont_img)
    
    # ガンマ変換(1<γ)　聖なる光に包まれます　可変パラメータはgamma1の0.1
    for m in range(1,c+1):
        
        # ガンマ変換ルックアップテーブル
        LUT_G1 = np.arange(256, dtype = 'uint8' )
        gamma1 = 1+0.1*m
        for i in range(256):
            LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
            
        #　代入
        gamma1_img = cv2.LUT(img, LUT_G1)
        
        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+m), gamma1_img)
        
    # ガンマ変換(1>γ)　ダークサイドへ堕ちます　可変パラメータはgamma2の0.1
    for m in range(1,d+1):
        
        # ガンマ変換ルックアップテーブル
        LUT_G2 = np.arange(256, dtype = 'uint8' )
        gamma2 = 1-0.1*m
        for i in range(256):
            LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
            
        #　代入
        gamma2_img = cv2.LUT(img, LUT_G2)
        
        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+c+m), gamma2_img)
    
    #　平滑化 引き伸ばします　可変パラメータはaverage_squareの10
    #        平滑化のみ引数が2つになっており、e*f 枚出力します
    for m in range(1,e+1):
        for n in range(1,f+1):
            average_square = (10*m,10*n)
            
            #　代入
            blur_img = cv2.blur(img,average_square)
            
            # 結果を出力
            cv2.imwrite("tokyo{}.jpg".format(a+b+c+d+5*(m-1)+n), blur_img)
            
    # ガウスノイズ付加　粗くなります　可変パラメータはsigmaの15
    for m in range(1,g+1):
        row,col,ch= img.shape
        mean = 0
        sigma = 15*m
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        gauss_img = img + gauss
        
        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+c+d+e*f+m), gauss_img)
        
    #　アフィン変換(左回転)　左回りに回転します　可変パラメータはangle1の10など
    for m in range(1,h+1):
    
        # 画像のサイズ取得
        size = tuple([img.shape[1], img.shape[0]])

        # 回転の中心座標（画像の中心）
        center = tuple([int(size[0]/2), int(size[1]/2)])

        # 回転角度・拡大率
        angle1, scale = 10*m, 1.0 

        # 回転行列の計算
        R1 = cv2.getRotationMatrix2D(center, angle1, scale) 

        # アフィン変換
        dst1 = cv2.warpAffine(img, R1, size, flags=cv2.INTER_CUBIC) 

        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+c+d+e*f+g+m), dst1)
        
    #　アフィン変換(右回転)　右回りに回転します　可変パラメータはangle2の10など
    for m in range(1,j+1):
    
        # 画像のサイズ取得
        size = tuple([img.shape[1], img.shape[0]])

        # 回転の中心座標（画像の中心）
        center = tuple([int(size[0]/2), int(size[1]/2)])

        # 回転角度・拡大率
        angle2, scale = -10*m, 1.0  

        # 回転行列の計算
        R2 = cv2.getRotationMatrix2D(center, angle2, scale) 

        # アフィン変換
        dst2 = cv2.warpAffine(img, R2, size, flags=cv2.INTER_CUBIC) 

        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+c+d+e*f+g+h+m), dst2)
    
    #　拡大
    for m in range(1,k+1):
        dst = cv2.resize(img,None,fx=1+0.1*m, fy=1+0.1*m, interpolation = cv2.INTER_CUBIC)
        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+c+d+e*f+g+h+m+j), dst)
    #　縮小
    for m in range(1,l+1):
        dst = cv2.resize(img,None,fx=1-0.1*m, fy=1-0.1*m, interpolation = cv2.INTER_CUBIC)
        # 結果を出力
        cv2.imwrite("tokyo{}.jpg".format(a+b+c+d+e*f+g+h+m+j+k), dst)


    
    
if __name__ == "__main__":
    main()