import numpy as np
import cv2 as cv

def process_frame(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 2)

    kernel = np.ones((3, 3), np.uint8)
    gray = cv.erode(gray, kernel, iterations=1)

    return True, gray