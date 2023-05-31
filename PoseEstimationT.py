import numpy as np
import cv2 as cv
import glob

NUM_COLS = 7
NUM_ROWS = 6

# draw XYZ axes
def draw(img, corners, imgpts):
    corner = tuple(map(int, corners[0].ravel()))
    print('\ncorner: ', corner)

    print('point 2: ', tuple(map(int, imgpts[0].ravel())))
    print('point 2: ', tuple(map(int, imgpts[1].ravel())))
    print('point 2: ', tuple(map(int, imgpts[2].ravel())))

    img = cv.line(img, corner, tuple(map(int, imgpts[0].ravel())), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0, 0, 255), 5)
    return img

# load in distoration parameters calculated earlier
with np.load('distortionparams.npz') as f:
    mtx, dist, _, _ = [f[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((NUM_COLS*NUM_ROWS,3), np.float32)
objp[:,:2] = np.mgrid[0:NUM_COLS,0:NUM_ROWS].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# record camera
cam = cv.VideoCapture(0)
while True:
    ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #ret, gray = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
    gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 2)

    kernel = np.ones((3, 3), np.uint8)
    gray = cv.erode(gray, kernel, iterations=1)

    cv.imshow('thresh', gray)

    ret, corners = cv.findChessboardCorners(gray, (NUM_COLS,NUM_ROWS), None)

    if ret:
        subcorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs = cv.solvePnP(objp, subcorners, mtx, dist)

        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, subcorners, imgpts)
    
    cv.imshow('img', img)
        
    k = cv.waitKey(1)
    if k == 27:
        break

cv.destroyAllWindows()