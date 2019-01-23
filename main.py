#!/usr/bin/env python
# -*- coding: utf-8 -*-

### CR - 20 (The Fake Ronaldo)

import numpy as np
import cv2
import collections
import serial
from picamera.array import PiRGBArray
from picamera import PiCamera
import time, math
import imutils

nondet = 10
whitemax = 0
lol = 4
lol2 = 5
greenmax = 0
val = False
cnt_blue = 0
blue_detected = False
blue_fre = True
frequency_map = {}
centres = []

def blur(img,kernel_size):
    ram = cv2.medianBlur(img,kernel_size,0)
    test = cv2.GaussianBlur(ram,(kernel_size,kernel_size),0)
    return test


def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img,low_threshold,high_threshold)


def extract_lines(img, lines, color=[255, 0, 0], thickness=2):

    # X cordinates of corresponding lane
    left_x = collections.defaultdict(list)
    right_x = collections.defaultdict(list)
    top_x = collections.defaultdict(list)

    # Y cordinates of corresponding lane
    left_y = collections.defaultdict(list)
    right_y = collections.defaultdict(list)
    top_y = collections.defaultdict(list)

    try:
        for line in lines:
            for x1,y1,x2,y2 in line:

                # Calculate slope
                slope = (y2-y1)*1.0/(x2-x1)

                # Grouping together slopes of variation 4
                # If abs(slope) is less than 20 deg it is in the top category
                if math.fabs(slope) < math.tan(np.pi/9):
                    top_x[int(math.atan(slope)*60/np.pi)].extend([x1,x2])
                    top_y[int(math.atan(slope)*60/np.pi)].extend([y1,y2])

                # If slope is less than -20 deg it is in the left category
                elif slope < math.tan(-np.pi/9) and slope > math.tan(-np.pi*4/9):
                    left_x[int(math.atan(slope)*60/np.pi)].extend([x1,x2])
                    left_y[int(math.atan(slope)*60/np.pi)].extend([y1,y2])

                # If slope is greater than 20 deg it is in the right category
                elif slope > math.tan(np.pi/9) and slope < math.tan(np.pi*4/9):
                    right_x[int(math.atan(slope)*60/np.pi)].extend([x1,x2])
                    right_y[int(math.atan(slope)*60/np.pi)].extend([y1,y2])
    except TypeError:
        pass
        
    max_y = img.shape[0]
    min_y = 0

    eqns = [None for i in range(3)]

    # Use the slope for the angle that has the maximum occurence and square fits
    # the points to get an approximate line equation that passes through all the point

    # Left
    try:
        _, left_slope = max((len(v),k) for k,v in left_x.items())
        lef_l = np.poly1d(np.polyfit(left_y[left_slope],left_x[left_slope],1))
        left_x_st = int(lef_l(max_y))
        left_x_en = int(lef_l(min_y))
        cv2.line(img,(left_x_st,max_y),(left_x_en,min_y),[255,0,0],thickness)
        eqns[0]=lef_l

    except:
        left_slope = None
        print("left ignored")

    # Right
    try:
        _, right_slope = max((len(v),k) for k,v in right_x.items())
        rig_l = np.poly1d(np.polyfit(right_y[right_slope],right_x[right_slope],1))
        right_x_st = int(rig_l(max_y))
        right_x_en = int(rig_l(min_y))
        cv2.line(img,(right_x_st,max_y),(right_x_en,min_y),[0,255,0],thickness)
        eqns[1]=rig_l

    except:
        right_slope = None
        print("right ignored")

    # Top
    try:
        _, top_slope = max((len(v),k) for k,v in top_x.items())
        top_l = np.poly1d(np.polyfit(top_y[top_slope],top_x[top_slope],1))
        top_x_st = int(top_l(max_y))
        top_x_en = int(top_l(min_y))
        cv2.line(img,(top_x_st,max_y),(top_x_en,min_y),[0,0,255],thickness)
        eqns[2]=top_l

    except:
        top_slope = None
        print("top ignored")    
    
    return eqns,img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
  
    # Extracts the hough lines
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    # An empty black image
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Returns the eqns and the image extracted
    return extract_lines(line_img, lines)


def find_move(eqns,img):

    """
    1 - Left
    1.5 - Forward Left
    2 - Right
    2.5 - Forward Right
    3 - Move forward
    4 - Reverse
    """

    # When Only Right Lane is detected
    if eqns[0] is None and eqns[1] is not None and eqns[2] is None:
        return 1

    # When only left is detected    
    if eqns[1] is None and eqns[0] is not None and eqns[2] is None:
        return 2

    # When right and top lane is detected    
    if eqns[0] is None and eqns[1] is not None and eqns[2] is not None:
        cnt = np.sum(img > 100)
        # When white pixel is less than 40
        if cnt*1.0/tot < 0.40:
            return 1.5
        else:
            return 3
    
    # When left and top lane is detected    
    if eqns[1] is None and eqns[0] is not None and eqns[2] is not None:
        cnt = np.sum(img > 100)
        # When white pixel is less than 40
        if cnt*1.0/tot < 0.40:
            return 2.5
        else:
            return 3

    # When no lines are there         
    if eqns[0] is None and eqns[1] is None and eqns[2] is None:
        return 4

    # When only top line is visible
    if eqns[0] is None and eqns[1] is None:
        a,b = np.hsplit(img,2)
        cnta = np.sum(a > 100)
        cntb = np.sum(b > 100)
        cnt = np.sum(img > 100)

        # If total pixel count is greater than 35%
        if cnt*1.0/tot > 0.35:
            return 3

        # If left side image has more white pixels
        elif cnta > cntb:
            return 1
        
        else:
            return 2
    return 3


def move_bot(board,num):
    
    if num == 2:
        print("RIGHT \n")
        board.write("D")
        time.sleep(0.2)
        board.write("B")
        return
        
    if num == 2.5:
        print("FORW RIGHT")
        board.write("W")
        time.sleep(0.2)
        board.write("B")
        time.sleep(0.2)
        board.write("D")
        time.sleep(0.2)
        board.write("B")
        return
    
    if num == 1.5:
        print("FORW LEFT")
        board.write("W")
        time.sleep(0.2)
        board.write("B")
        time.sleep(0.2)
        board.write("A")
        time.sleep(0.2)
        board.write("B")
        return    

    if num == 1:
        print("LEFT \n")
        board.write("A")
        time.sleep(0.2)
        board.write("B")
        return

    if num == 3:
        print("FORWARD \n")
        board.write("W")
        time.sleep(0.2)
        board.write("B")
        return

    if num == 4:
        print("REVERSE \n")
        board.write("R")
        time.sleep(0.4)
        board.write("B")
        return

    if num == 5:
        board.write("B")
        time.sleep(0.5)
        board.write("L")
        time.sleep(1)
        board.write("U")
        return

# Detects the ending green floor
def end_detect(img):

    global greenmax
    lower = np.array([50,110,90],dtype="uint8")
    upper = np.array([95,170,150],dtype = "uint8")
    print(img[img.shape[0]/2,img.shape[1]/2])
    mask = cv2.inRange(img,lower,upper)
    white  = np.sum(mask > 100)    

    if white > greenmax:
        greenmax = white

    print('Green = ',white)
    if greenmax > 1200 and white < 20:
        greenmax=0
        return True
    else:
        return False
    
# Detects the red led    
def led_detect(img):
    
    global nondet
    global whitemax
    lower = np.array([0,0,80],dtype="uint8")
    upper = np.array([70,70,230],dtype = "uint8")

    mask = cv2.inRange(img,lower,upper)
    white = np.sum(mask > 100)    
    
    if nondet < 4:
        nondet += 1
        return False	
    
    if white > whitemax:
        whitemax = white
        
    print("Red = ",white)
    if white < 50 and whitemax > 200:
        nondet = 0
        whitemax = 0
        return True
    else :
        return False

# Centre of the contour
def find_centre(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [cX,cY] 

# Finds whether two blue blobs are present are not
def detect_blue(img):

    lower = np.array([80,0,0],dtype="uint8")
    upper = np.array([255,70,90],dtype = "uint8")

    mask = cv2.inRange(img,lower,upper)
    blue = np.sum(mask > 100)

    if blue < 2000:
        return False

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
	        cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) < 2:
        return False

    for ind,countor in enumerate(cnts):
        centres.append(find_centre(countor))
        frequency_map[ind] = 0
    
    return True


# Returns distance
def dist(a,b,c,d):
    return (c-a)*(c-a)+(d-b)*(d-b)


def detect_freq(images):

    for img in images:

    # Get the binary
        lower = np.array([80,0,0],dtype="uint8")
        upper = np.array([255,70,90],dtype = "uint8")

        mask = cv2.inRange(img,lower,upper)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                            	cv2.CHAIN_APPROX_SIMPLE)


    # If the corresponding blob is present in the image, increase its occurence value            
        for c in cnts:
            cen = find_centre(c)
            temp=10**9
            val = -1
            for ind,centre in enumerate(centres):
                if dist(cen[0],cen[1],centre[0],centre[1]) < temp:
                    temp =  dist(cen[0],cen[1],centre[0],centre[1])
                    val = ind
            
            frequency_map[val] += 1

    temp = -1
    temp_low = 10**9
    highest = lowest = None

    for k,v in frequency_map.items():
        if temp < v:
            temp = v
            highest = k
        if temp_low > v:
            temp_low = v
            lowest = k

    # Moves towards higher blinking frequency
    if centres[highest][0] < centres[lowest][0]:
        move_bot(board,1.5)

    else:
        move_bot(board,2.5)  


if __name__ == '__main__':

    # Initialising serial communication
    global board
    board = serial.Serial("/dev/ttyACM0",9600,timeout=1)
    time.sleep(1)

    # Initialising video feed
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 12
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # Camera warmup time
    time.sleep(1)

    global tot
    tot = 640*480
    images = []

    # Accessing the frames
    for frame in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
        
        img = frame.array

        ############################ PATH LED ###############################    
        
        if not blue_detected:
            if not val:
                val = detect_blue(img)
            
            if val:
                cnt_blue+=1
                images.append(img)
                rawCapture.truncate(0)
                if cnt_blue > 48:
                    blue_detected = True
                    val = False
                    cnt_blue = 0
                continue

        if blue_detected and blue_fre:
            blue_fre = False
            detect_freq(images)
            rawCapture.truncate(0)
            continue
        ######################################################################


        
        ############################ WALL LED ###############################    
        # Returns true if led disappears
        led = led_detect(img)

        if led:
            lol = 0

        lol += 1
        # After 3 moves after detecting LED, blink    
        if lol == 3:
                move_bot(board,5)
        ######################################################################

        ############################# END DETECT ###############################
        end  = end_detect(img)
        if end:
            lol2=0

        lol2+=1

        if lol2 == 2:
            # move_bot(board,5)
            board.write("B")
            print("Reached Successfully")
            exit(0)
        ############################# END DETECT ##############################
        

        # Grey Scale
        gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Blurring
        gauss_gray = blur(gray_image,3)

        # Binary
        ran_start = 70
        mask_white = cv2.inRange(gauss_gray,ran_start,255)
        
        # Canny
        low_threshold = 100
        high_threshold = 300
        canny_edges = canny(gauss_gray,low_threshold,high_threshold)

        # Hough
        threshold = 50
        rho = 1
        theta = np.pi/180
        min_line_len = 60
        max_line_gap = 10
        
        eqns, hough_img = hough_lines(canny_edges,rho,theta,threshold,min_line_len,max_line_gap)
       
        # Determining moves
        num = find_move(eqns,mask_white)

        # End, if taking reverse
        if lol2 < 2 and num == 4:
            board.write("B")
            print("Reached Successfully")
            exit(0)

        # Move the bot
        move_bot(board,num)

        # Ready for next frame
        rawCapture.truncate(0)

        time.sleep(0.15)
        
        #cv2.namedWindow("hough",cv2.WINDOW_NORMAL)
        #cv2.namedWindow("canny",cv2.WINDOW_NORMAL)
        #cv2.namedWindow("origin",cv2.WINDOW_NORMAL)
        #cv2.namedWindow("maskWhite",cv2.WINDOW_NORMAL)

        #cv2.imshow("hough",hough_img)
        #cv2.imshow("canny",canny_edges)
        #cv2.imshow("origin",img)
        #cv2.imshow("maskWhite",mask_white)

        #cv2.waitKey(1000)







