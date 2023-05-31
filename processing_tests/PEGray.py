import numpy as np
import cv2 as cv

def process_frame(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray