import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
#left_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rimg=cv2.flip(gray,1)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    profilesLeft = profile_cascade.detectMultiScale(gray, 1.3, 5)
    profilesRight = profile_cascade.detectMultiScale(rimg, 1.3, 5)

    for (x,y,w,h) in profilesRight:
        cv2.rectangle(img, (img.shape[1] - x,y),(img.shape[1]-x-w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        #left = left_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in left:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,180,180),2)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,250,0),2)

    for (x,y,w,h) in profilesLeft:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        #left = left_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in left:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,180,180),2)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,250,0),2)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            #cv2.circle(roi_color,(ex+int(ew/2),ey+int(eh/2)),int(ew/2),(255,255,255),20)
            #cv2.circle(roi_color,(ex+int(ew/2),ey+int(eh/2)),int(ew/5),(0,0,0),20)
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #smiles = smile_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in smiles:
        #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
