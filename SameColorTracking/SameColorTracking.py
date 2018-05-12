import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, img = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    #lower_yellow = np.array([23, 59, 119], np.uint8)
    #upper_yellow = np.array([54,255,255], np.uint8)

    lower_yellow = np.array([33, 99, 159], np.uint8)
    upper_yellow = np.array([54,255,255], np.uint8)


    yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernal = np.ones((5 ,5), "uint8")

    yellow = cv2.dilate(yellow, kernal)

    (_,contours, hierarchy)=cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = 0;
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area>300):
            x,y,w,h = cv2.boundingRect(contour)

            if w > 5:
                n += 1
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                img = cv2.circle(img, (x+int(w/2),y+int(h/2)), int(w/2), (0,255,0), 2)
                cv2.putText(img,"Ball:"+str(n)+" X="+str(x)+" Y="+str(y),(10,30+(15*n)),cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))

    cv2.imshow("Color Tracking", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break



