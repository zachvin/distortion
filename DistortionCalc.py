import cv2 as cv
import numpy as np
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((6*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# arrays to store object poitns and image points
objpoints = []
imgpoints = []

images = glob.glob('./calib/*.jpg')
good_images = []

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    # refine points and add (should run on all photos, i.e. all photos should have successful recognition)
    if ret:
        good_images.append(fname)
        objpoints.append(objp)

        subcorners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(subcorners)

        # draw and display corners
        cv.drawChessboardCorners(img, (7,6), subcorners, ret)
        cv.imshow('img', img)
        k = cv.waitKey(500)
        if k == 27:
            break
    else:
        print(f'ERROR: image {fname} not detected')

cv.destroyAllWindows()

# calibrate
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# undistort
fname = './calib/testdistort.jpg'
img = cv.imread(fname)
h,w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('./calib/calibresult.jpg', dst)

cv.imshow('Original', img)
cv.imshow('Undistorted', dst)

while True:
    k = cv.waitKey(500)
    if k == 27:
        break

cv.destroyAllWindows()