import cv2
from cv2 import COLOR_GRAY2BGR
from cv2 import MORPH_RECT
from cv2 import Mat
import numpy as np



if __name__ == '__main__':
    path = 'traffic.mp4'
    cap = cv2.VideoCapture(path)

    ret,frame = cap.read()

    rows,cols,channel = frame.shape


    preprocessing = []

    oldFrames = []

    meanFrame = frame.copy()

    kernel = np.ones((5,5), np.uint8)
    
    variance = 15
    while(cap.isOpened()):  
        
        foreground = frame.copy()

        for r in range(0,rows):
            for c in range(0,cols):
                if(abs(frame[r,c,0] - meanFrame[r,c,0]) < variance or abs(frame[r,c,1] - meanFrame[r,c,1]) < variance or abs(frame[r,c,2] - meanFrame[r,c,2]) < variance):
                    foreground[r,c,0] = 0
                    foreground[r,c,1] = 0
                    foreground[r,c,2] = 0

        #// preprocessing
        preprocessing.clear()
        preprocessingLast : Mat = cv2.cvtColor(foreground,cv2.COLOR_BGR2GRAY)
        preprocessing.append(preprocessingLast)
        preprocessingLast : Mat =cv2.dilate(preprocessing.pop(),kernel,iterations=1)
        preprocessing.append(preprocessingLast)
        preprocessingLast : Mat = cv2.erode(preprocessing.pop(),kernel)
        preprocessing.append(preprocessingLast)
        preprocessingLast : Mat =cv2.medianBlur(preprocessing.pop(),3)
        preprocessing.append(preprocessingLast)
        # show the new frame and the preprocessed frame

        if ret == True:
            cv2.imshow('video',frame)
            cv2.imshow('foreground',preprocessingLast)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # make the mean frame a rolling average

        meanFrame = frame.copy()
        
        ret,frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()



    
