import cv2
import numpy as np
import matplotlib.pyplot as plt
imge = cv2.imread("photo.png") # rad the photo 
gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY) # confort to 


corner = cv2.goodFeaturesToTrack(gray , 30,0.01, 10 ) # gray = var up , 30 = how many dot , 0.01 engil , 10 qulaty effact 
corner = np.int0(corner)

for i in corner :
        x, y = i.ravel()
        cv2.circle(imge , (x, y) ,3 ,255 , -1 )  #  3 = readis , 255 = red color , 
plt.imshow(imge) # show ruslt 
plt.waitforbuttonpress() # to display until press close 