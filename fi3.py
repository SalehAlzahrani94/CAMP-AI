import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests# import for requests form onlin
from PIL import Image   # to read onlin 
# read form onlin sorce 
imge = Image.open(requests.get('https://a57.foxnews.com/media.foxbusiness.com/BrightCove/854081161001/201805/2879/931/524/854081161001_5782482890001_5782477388001-vs.jpg',stream=True).raw)
imge = imge.resize((450,250)) # change size of ohpht

# make photo clear to AI 
imge_array = np.array(imge) # confert to matix 
gray = cv2.cvtColor(imge_array,cv2.COLOR_BGR2GRAY) # change color to gray color  (to cv can read easy )
blur = cv2.GaussianBlur(gray,(5,5), 0) # remove nose , number not chcnge 
Image.fromarray(blur) # توسع الصورة وتزيد الاماكن البيضاء 
dilated = cv2.dilate(blur,np.ones((3,3)))
Image.fromarray(dilated) # توسع الصورة وتزيد الاماكن البيضاء 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))  # to make easer to user 
closing = cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel)
Image.fromarray(closing)

# take data 
car_src = 'haarcascade_car.xml' # data ready form Database given (tran to identfy cars )
car = cv2.CascadeClassifier(car_src)
cars = car.detectMultiScale(closing,1.1,1) # now AI know cars ( idintfy )

# make reqtancle for each car
cnt = 0
for (x,y,w,h) in cars:
    cv2.rectangle(imge_array,(x,y),(x+w,y+h),(255,0,0),2) # draw rectangle in car
    cnt += 1 # count cars 
    print("number of cars " ,cnt) # show car numbers 
    Image.fromarray(imge_array)

cv2.imshow('image',imge_array)
cv2.waitKey()
cv2.destroyAllWindows()

