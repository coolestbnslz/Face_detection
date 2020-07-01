import cv2
import numpy as np
image = cv2.imread('C:\\Users\\Nikhil Bansal\\Downloads\\myfirst.png')
#cv2.imshow("face",image)
#cv2.waitKey(0)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gary',gray)
#cv2.waitKey(0)
facecasacde=cv2.CascadeClassifier('C:\\Users\\Nikhil Bansal\\Downloads\\face.xml')
faces=facecasacde.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))
print(faces)
for(x,y,w,h) in faces:
    cv2.putText(image,'Face 1',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),4)
#cv2.imshow('face',image)
#cv2.waitKey(0)
#webcam=cv2.VideoCapture(0)
#ret,frame=webcam.read()
#print(ret)
#webcam.release()
#cv2.imshow('my image',frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
webcam=cv2.VideoCapture(0)
while(True):
    ret,frame=webcam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = facecasacde.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in faces:
        #cv2.putText(image, 'Face 1', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('gray',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF==ord('a'):
       cv2.imwrite("C:\\Users\\Nikhil Bansal\\Desktop\\myimage.jpg", frame)

webcam.release()
cv2.destroyAllWindows()

