import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') # read database 
camera = cv2.VideoCapture(0) # 0  open inner camera , 1 outer camera

while (True):
    read_ok, frame = camera.read() # frame by frame nd chceh its workd
    leabels = []
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) # vhange color
    faces = face.detectMultiScale(gray) # find faces

    for (x,y,w,h) in faces: # draw rectangle 
        cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,255,0),5)

    cv2.imshow('frame' , frame) # display 
    if cv2.waitKey(1) & 0xFF == ord('s'): #  prass s to stpe
        break
camera.release()
cv2.destroyAllWindows()
