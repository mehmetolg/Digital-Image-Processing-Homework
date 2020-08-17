import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

orjinal = cv.imread('./196073.jpg')
img = cv.imread('./196073.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()

img = cv.bilateralFilter(img,5,70,70)
cv.imshow('bilateral filtresi',img)
cv.waitKey(0)

a,img=cv.threshold(img, 153, 255, cv.THRESH_BINARY)

cv.imshow('siyah beyaz',img)
cv.waitKey(0)

cizgi =np.ones((3,3),np.uint8)

kernelDilationXv1 = np.ones((3,3),np.uint8)
kernelDilationXv1[0,1]=0
kernelDilationXv1[0,2]=0
kernelDilationXv1[1,0]=0
kernelDilationXv1[1,2]=0
kernelDilationXv1[2,0]=0
kernelDilationXv1[2,1]=0

kernelDilationX = np.ones((1,5),np.uint8)

kernelDilationY = np.ones((5,1),np.uint8)

img = cv.dilate(img,kernelDilationY,iterations = 2)
cv.imshow('genisletme sonrasi y',img)
cv.waitKey(0)
img = cv.dilate(img,kernelDilationX,iterations = 2)
cv.imshow('genisletme sonrasi X',img)
cv.waitKey(0)

img = cv.dilate(img,kernelDilationXv1,iterations = 2)
cv.imshow('genisletme sonrasi x2',img)
cv.waitKey(0)

renkli = np.ndarray((img.shape[0],img.shape[1],3),dtype=np.uint8)
renkli.fill(0)
renkli[:,:,1] = renkli[:,:,1]+img
renkli = cv.morphologyEx(renkli, cv.MORPH_GRADIENT, cizgi,iterations=1)
cv.imshow('son',renkli)
cv.waitKey(0)
for row in range(renkli.shape[0]):
    for column in range (renkli.shape[1]):
        if(renkli[row,column,1]>128):
            orjinal[row,column,0] = 0
            orjinal[row,column,1] = 255
            orjinal[row,column,2] = 0


cv.imshow('son',orjinal)
cv.waitKey(0)
cv.imwrite('./son.jpg',orjinal)
