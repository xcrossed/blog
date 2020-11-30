import os

import cv2 as cv
import numpy as np


#原理
# y=(f(x)/4)*4 +w/4
# srcPath表示源图片路径
# waterMarkPath表示水印路径
# dstPath　表示加完水印后的照片路径
def encode(srcPath,waterMarkPath,dstPath):
    srcImg=cv.imread(srcPath)
    waterMarkImg=cv.imread(waterMarkPath)

    print(srcImg.shape)
    print(waterMarkImg.shape)
    srcImg=srcImg.astype(np.uint8)
    #将源图片低2位置0
    for x in range(waterMarkImg.shape[0]):
        for y in range(waterMarkImg.shape[1]):
            srcImg[x][y] =srcImg[x][y] & 0b11111000 #原图低２位置0
            srcImg[x][y] += (waterMarkImg[x][y] >> 5) #水印右移６位，只保留高位

    cv.imwrite(dstPath,srcImg)
# srcPath 表示包含水印的图片路径
# extractWaterMarkPath　表示提取的水印路径
# 将高６位置０，将低２位左移６位，即得到水印
def decode(srcPath,watextractWaterMarkPath):
    srcImg=cv.imread(srcPath)
    waterMarkImg=srcImg.copy().astype(np.uint8)
    for x in range(waterMarkImg.shape[0]):
        for y in range(waterMarkImg.shape[1]):
            waterMarkImg[x][y] =waterMarkImg[x][y] & 0b00000111 #高６位置０，保留低２位
            waterMarkImg[x][y] =waterMarkImg[x][y] << 5
    cv.imwrite(watextractWaterMarkPath,waterMarkImg)

if __name__ == "__main__":
    basePath=r"/data/git/blog/code/imageProcess"
    srcPath=os.path.join(basePath,r"./input/src.jpeg")
    waterMarkPath=os.path.join(basePath,r"./input/logo.jpeg")
    dstPath=os.path.join(basePath,r"./output/dstWithWarterMark.jpeg")
    extractWaterMarkPath=os.path.join(basePath,r"./output/extractWaterMark.jpeg")
    encode(srcPath,waterMarkPath,dstPath)
    decode(dstPath,extractWaterMarkPath)
