import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os

path = os.getcwd() + "/images/Capoff/"
img_path =  path + "img973.jpg"

img = cv2.imread(img_path)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

_,thresh = cv2.threshold(imgray,90,255,cv2.THRESH_BINARY)


cv2.imshow('gray',thresh)
cv2.waitKey(0)
#plt.hist(imgray.ravel(),bins = 1000)

#plt.show()image = np.array([[1, 0, 0, 1], [0, 0, 0,0], [0, 0, 0,0], [1, 0, 0,1]])