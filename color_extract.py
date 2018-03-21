import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os

path = os.getcwd() + "/../Dataset/train/"
img_path =  path + "img920.jpg"
print(img_path)
img = cv2.imread(img_path)
imhsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
 
cv2.imshow('hsv',imhsv)


lower_blue = np.array([110,50,50])
upper_blue = np.array([130,50,50])
blue_mask = cv2.inRange(imhsv, lower_blue, upper_blue)
blue_pixels = sum(map(sum,blue_mask))
blue_area_covered = blue_pixels / (blue_mask.shape[0] * blue_mask.shape[1])
print(blue_area_covered)

lower_green = np.array([30,50,50])
upper_green = np.array([40,255,255])	
green_mask = cv2.inRange(imhsv, lower_green, upper_green)
green_pixels = sum(map(sum,green_mask))

green_area_covered = green_pixels / (green_mask.shape[0] * green_mask.shape[1])
print(green_area_covered)



cv2.imshow('green',green_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


