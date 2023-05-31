import cv2 as cv
import numpy as np
import PEGaussian as uut

NUM_COLS = 7
NUM_ROWS = 6

with np.load('distortionparams.npz') as f:
    mtx, dist, _, _ = [f[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((NUM_COLS*NUM_ROWS,3), np.float32)
objp[:,:2] = np.mgrid[0:NUM_COLS,0:NUM_ROWS].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

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

vid = cv.VideoCapture(0)

current_frame = 0
frames_identified = 0
while True:
    ret, img = vid.read()    
    if not ret:
        break

    chessfound = False
    frame = uut.process_frame(img)
    chessfound, corners = cv.findChessboardCorners(frame, (7,6), None)

    if chessfound:
        frames_identified += 1
        subcorners = cv.cornerSubPix(frame, corners, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs = cv.solvePnP(objp, subcorners, mtx, dist)

        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, subcorners, imgpts)

    cv.imshow('Video frame', img)
    cv.imshow('UUT frame', frame)

    current_frame += 1

    k = cv.waitKey(1)
    if k == 27:
        break

print(f'SUMMARY: {frames_identified} out of {current_frame} frames correctly identified ({frames_identified/current_frame*100:.2f}%)')

vid.release()
cv.destroyAllWindows()