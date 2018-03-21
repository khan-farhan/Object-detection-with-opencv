import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os
import pandas as pd



def imshow(path,name,waitKey = False):

 	cv2.imshow(name,path)

 	if waitKey:

 		cv2.waitKey(0)

 	return

def contour(img):
	_,cnts,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	return cnts



def MorphOP(img,name,kernel,iterations = 1):

	if name == "erosion":

		img = cv2.erode(img,kernel,iterations = 1)

	elif name == "dilate":

		img = cv2.dilate(img,kernel,iterations = 1)

	elif name == "open":

		img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

	elif name == "close":

		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	else:

		print("Choose an appropriate MorphOP")

	return img


def getKernel(name, size):

	if name == "rectangular":

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,size)

	elif name == "ellipse":

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)

	elif name == "cross":

		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,size)

	else:

		print("Enter appropriate kernel structure")

		return 

	return kernel



def clustering(img,K,criteria):

	Z = img.reshape((-1,3))
	Z = np.float32(Z)
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)
	res = center[label.flatten()]
	clustered = res.reshape((img.shape))

	return label,clustered,center

def foo(cnt):
	[_,_,w,h] = cv2.boundingRect(cnt)
	area = w*h
	return area


capoff_path = os.getcwd() + "/images/Capoff/"
capon_path = os.getcwd() + "/images/Capon/"
labeloncapoff_path = os.getcwd() + "/images/labeloncapoff/"
labeloncapon_path = os.getcwd() + "/images/labeloncapon/"

csv_path = os.getcwd() + "/images/labelonCapoff.csv"
df = pd.read_csv(csv_path)


"""
capofflabeloff = 0
caponlabeloff = 1
capofflabelon = 2
caponlabelon = 3

"""

if __name__ == '__main__':

		k = 4
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

		os.chdir(labeloncapoff_path)

		count = 0

		for img,label in zip(df.image,df.label):

			print(img)
			image = cv2.imread(img)

			result = ""

			_,clustered,_ = clustering(image,k,criteria)
			
			blur = cv2.GaussianBlur(clustered,(5,5),0)
			
			edges = cv2.Canny(blur, 50, 200, 255,L2gradient= True)
			
			kernel = getKernel("rectangular",(5,5))
			
			dilated = MorphOP(edges,"dilate",kernel)
			
			cnts = contour(dilated)
			cnts = sorted(cnts,key = foo,reverse = True)


			if len(cnts) == 0:
				
				result = 0

			elif len(cnts) == 1:
		
				[x,y,w,h] = cv2.boundingRect(cnts[0])
				AreaROI = w*h

				if AreaROI > 18000:
					result = 2

				elif AreaROI > 5000:

					result = 1

				else:
					result = 0

			elif len(cnts) == 2:
				
				[x,y,w,h] = cv2.boundingRect(cnts[0])
				[_,_,w1,h1] = cv2.boundingRect(cnts[1])
				AreaROI = w*h
				AreaROI1 = w1*h1 
				#print(AreaROI1,AreaROI)

				if AreaROI > 18000 and AreaROI1 > 18000:				## overlaping label contours
					result = 2

				elif AreaROI > 18000 and AreaROI1 > 5000 and AreaROI1 < 18000:
					result = 3

				elif AreaROI > 18000 and AreaROI1 < 5000:
					result = 2


				elif AreaROI > 5000 and AreaROI < 18000:
					result = 1

				else:
					result = 0

			else:
				
				[x,y,w,h] = cv2.boundingRect(cnts[0])
				[_,_,w1,h1] = cv2.boundingRect(cnts[1])
				[_,_,w2,h2] = cv2.boundingRect(cnts[2])

				AreaROI = w*h
				AreaROI1 = w1*h1
				AreaROI2 = w2*h2

				if AreaROI > 18000 and AreaROI1 > 5000 and AreaROI1 < 18000: 						
					result = 3

				if AreaROI > 18000 and AreaROI1 > 18000:							## overlaping label countours
					
					if AreaROI2 > 5000 and AreaROI2 < 18000:
						result = 3

					else:
						result = 2


				elif AreaROI > 5000 and AreaROI < 18000:
					result = 1

				else:
					result = 0


				image = edges[y:y+h,x:x+w]

				imshow(image,"extracted",waitKey = True)

			print(result)
			if result == label:
				count += 1


		Accuracy = count / df.shape[0]

		print("Accuracy obtained {}".format(Accuracy))






