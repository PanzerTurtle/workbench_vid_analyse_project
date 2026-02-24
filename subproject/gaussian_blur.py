import cv2

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while True:
    ret, frame2 = cap.read()
    
    # Process current frame
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    
    # Difference and Threshold
    diff = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate for better contour detection
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours (outlines)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500: # Minimum size threshold
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    cv2.imshow('Motion Detection', frame2)
    
    # Tick to next frame comparison
    gray1 = gray2
    
    cv2.waitKey(30) & 0xff
    if cv2.getWindowProperty('Motion Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
