import cv2
import numpy as np
import matplotlib.pyplot as plt
imge = cv2.imread("photo.png") # rad the photo 
gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY) # confort to 
gray = np.float32(gray) # confort to matix 

#with shapes ( tiangle)
des = cv2.cornerHarris(gray , 2,5,0,0.07)
des = cv2.dilate(des,None)
imge[des >0.01 * des.max()] = [255,0,0]


plt.imshow(imge) # show ruslt 
plt.waitforbuttonpress() # to display until press close 