import cv2

# cv2.aruco.DICT_4X4_50

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()

    (corners, ids, rejected) = arucoDetector.detectMarkers(img)

    if corners:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners,ids):
            corners = markerCorner.reshape((4, 2))
            (tL, tR, bR, bL) = corners

            tR = (int(tR[0]), int(tR[1]))
            tL = (int(tL[0]), int(tL[1]))
            bR = (int(bR[0]), int(bR[1]))
            bL = (int(bL[0]), int(bL[1]))

            cv2.line(img, tL, tR, (0, 255, 0), 2)
            cv2.line(img, tR, bR, (0, 255, 0), 2)
            cv2.line(img, bR, bL, (0, 255, 0), 2)
            cv2.line(img, bL, tL, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows
cap.close()
            