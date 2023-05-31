import cv2 as cv
import numpy as np

# CONSTANTS
RED_LOWER1 = np.array([0, 100, 20])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 100, 20])
RED_UPPER2 = np.array([180, 255, 255])

PADDING = 10

def process_frame(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    frame_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    frame_hsv = cv.GaussianBlur(frame_hsv, (11,11), 5)

    lower_mask = cv.inRange(frame_hsv, RED_LOWER1, RED_UPPER1)
    upper_mask = cv.inRange(frame_hsv, RED_LOWER2, RED_UPPER2)

    mask = lower_mask + upper_mask

    res = cv.bitwise_and(img, img, mask=mask)


    # find contours in mask
    contours,hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv.contourArea)
        x,y,w,h = cv.boundingRect(max_contour)

        mask = np.zeros(img.shape[:2], np.uint8)
        mask[y-PADDING:y+h+PADDING,x-PADDING:x+w+PADDING] = 255

        gray = cv.bitwise_and(gray,gray,mask=mask)

        gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 2)

        kernel = np.ones((3, 3), np.uint8)
        gray = cv.erode(gray, kernel, iterations=1)

        cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv.imshow('Bounding contours', img)
        #cv.drawContours(img, [max_contour], -1, (0,255,0), 3)

        return True, gray

    return False, gray