import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests# import for requests form onlin
from PIL import Image   # to read onlin 
#calls vidow and data
face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cape = cv2.VideoCapture('vid2.mp4')


while True :  # convorte to frame be fream 
    _,img = cape.read() # captur one fram 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert fram to gray 
    faces = face.detectMultiScale(gray,1.1,4) # find face in it and return cordintae x,y ,w,h

    for (x,y,w,h) in faces :  # draw reectangle each face 
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5) 
    cv2.imshow('vid',img) # display 

    # press esc key potin to stop
    k=cv2.waitKey(30) & 0xff
    if k == 27: # press esc key potin to stop
        break
cape.release()
