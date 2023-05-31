import numpy as np
import cv2 as cv
import glob

def draw(img, corners, imgpts):
    corner = tuple(map(int, corners[0].ravel()))
    print('imgpts: ', imgpts)
    print('corner: ', corner)

    print('point 2: ', tuple(map(int, imgpts[0].ravel())))
    print('point 2: ', tuple(map(int, imgpts[1].ravel())))
    print('point 2: ', tuple(map(int, imgpts[2].ravel())))

    img = cv.line(img, corner, tuple(map(int, imgpts[0].ravel())), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0, 0, 255), 5)
    return img

with np.load('distortionparams.npz') as f:
    mtx, dist, _, _ = [f[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

cam = cv.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

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