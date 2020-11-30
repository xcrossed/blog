import cv2
import numpy as np

from imageProcess.tools import base, t

if __name__ == "__main__":
    inputFile="./code/imageProcess/m50m97gfez.png"
    srcImg=cv2.imread(inputFile)
    gray=cv2.cvtColor(srcImg,cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3) 
    # dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])

    c1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 8))  
    c2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))

    ret, bimg = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)  
    dilation = cv2.dilate(bimg, c2, iterations=1)

    erosion = cv2.erode(dilation, c1, iterations=1)  
    img_edge = cv2.dilate(erosion, c2, iterations=1)  
    t.imgShow(img_edge)

    # contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  

    # # 记录文字区块数量
    # area_text_num = 0  
    # region = []

    
    # for i in range(len(contours)):  
    #     cnt = contours[i]
    #     area = cv2.contourArea(cnt)

    #     # 筛掉面积过小的区块
    #     if area < 1000:
    #         continue

    #     # 得到最小矩形区域，转换为顶点坐标形式（矩形可能会有角度）
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect)
    #     box = np.asarray(box)
    #     box = box.astype(int)

    #     x0 = box[0][0] if box[0][0] > 0 else 0
    #     x1 = box[2][0] if box[2][0] > 0 else 0
    #     y0 = box[0][1] if box[0][1] > 0 else 0
    #     y1 = box[2][1] if box[2][1] > 0 else 0
    #     height = abs(y0 - y1)
    #     width = abs(x0 - x1)

    #     # 筛掉不够“扁”的的区块，它们更有可能不是文字
    #     if height > width * 0.3:
    #         continue
    #     area_text_num += height * width
    #     region.append(box)

    # print( region, area_text_num  )
