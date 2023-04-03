import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') # read database 
camera = cv2.VideoCapture(0) # 0  open inner camera , 1 outer camera

while (True):
    read_ok, frame = camera.read() # frame by frame nd chceh its workd
    cv2.imshow('frame' , frame)
    if cv2.waitKey(1) & 0xFF == ord('s'): # if prass s to stpe
        break

camera.release()
cv2.destroyAllWindows()
