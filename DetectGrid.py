import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)

cv.namedWindow('Frame')
cv.namedWindow('Thresh')

while True:
    ret, frame = cam.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    if ret:
        cv.drawChessboardCorners(frame, (7,6), corners, ret)

    cv.imshow('Frame', frame)

    k = cv.waitKey(1)
    if k == 27:
        break

cv.destroyAllWindows()