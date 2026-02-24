import cv2 

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml') 

upperbody_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_upperbody.xml') 

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read() 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray) 

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
            
        bodies = upperbody_cascade.detectMultiScale(gray, 1.5, 5)
        for (bx,by,bw,bh) in bodies:
            cv2.rectangle(img,(bx,by),(bx+bw,by+bh),(255,255,0),2)

    cv2.imshow('face detection',img)

    cv2.waitKey(30) & 0xff
    if cv2.getWindowProperty('face detection', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
