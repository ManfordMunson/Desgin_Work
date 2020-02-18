from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
#Adds arguments for running a video file and sets buffer size with a default of 64

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
#****BGR based color not RGB****
#Creates bounds for darker and lighter of each color
green = True

redLower = (9, 31, 117)
redUpper = (139, 163, 255)
red = False

blueLower = (127, 48, 2)
blueUpper = (255, 175, 128)
blue = False
#If we set a color to true all others must be false 

pts = deque(maxlen=args['buffer'])

if not args.get('video', False):
    vs = VideoStream(src=0).start()

else:
    vs = cv2.VideoCapture(args['video'])

time.sleep(2.0)
#give time for camera to start

while True:
    frame = vs.read()
    frame = frame[1] if args.get('video',False) else frame
    
    if frame is None:
        break
    
    frame = imutils.resize(frame, width = 600)
    blurred = cv2.GaussianBlur(frame, (11,11),0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #blurs the image to get rid of noise 
    
    if blue:
        mask = cv2.inRange(hsv, blueLower, blueUpper)
    elif green:
        mask = cv2.inRange(hsv, greenLower, greenUpper)
    elif red:
        mask = cv2.inRange(hsv, redLower, redUpper)
    #color selection 
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #finds contours in image based on the mask based on color
    center = None
    
    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        ((x,y),radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        #finds center of circle to get midpoint
        
        if radius>10:
            cv2.circle(frame, (int(x),int(y)), int(radius),(0,255,255),2)
            cv2.circle(frame, center, 5, (0,0,255), -1)


    pts.appendleft(center)
    
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        
        thickness = int(np.sqrt(args['buffer']/float(i+1))*2.5)
        cv2.line(frame, pts[i-1], pts[i], (0,0,255), thickness)
        
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        #press q to close stream
        break

if not args.get('video', False):
    vs.stop()

else:
    vs.release()
    
cv2.destroyAllWindows()
