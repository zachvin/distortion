import cv2 as cv
import numpy as np
import glob

scale = 0.2

#ret, frame = cam.read()
images = glob.glob('./grid_sony/hallway-*.JPG')
for fname in images:
    print(fname)
    frame = cv.imread(fname)

    dim = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    print(dim)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.blur(gray, (5,5))

    ret, corners = cv.findChessboardCorners(gray, (4,4), None)

    if ret:
        cv.drawChessboardCorners(frame, (4,4), corners, ret)
        print('\tfound')

    resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    cv.imshow('Frame', resized)

    k = cv.waitKey(0)
    if k == 27:
        break

cv.destroyAllWindows()