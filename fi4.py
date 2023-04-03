import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests# import for requests form onlin
from PIL import Image   # to read onlin 

image = cv2.imread('seld2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray'), plt.show


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
face = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml") # read database 
face_c = face.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);# tAKE AND COUNT FACES IN PHOTH . note there is ; 

print('Faces found: ', len(face_c))

for(x_face, y_face, w_face, h_face) in face_c:  # draw reactangle each face 
    cv2.rectangle(image, (x_face, y_face), (x_face + w_face, y_face + w_face), (0, 255, 0), 5) # 5 for the size fo rectangle 
plt.imshow(convertToRGB(image)),plt.show()

