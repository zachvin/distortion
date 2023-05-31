import cv2 as cv

cam = cv.VideoCapture(0)

(w, h) = map(int, [cam.get(3), cam.get(4)])
recording = cv.VideoWriter('stresstest.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, (w,h))

while True:
    ret, img = cam.read()

    if ret:
        cv.imshow('img', img)
        recording.write(img)


    k = cv.waitKey(1)
    if k == 27:
        break


recording.release()
cam.release()
cv.destroyAllWindows()