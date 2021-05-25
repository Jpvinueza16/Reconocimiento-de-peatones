
from imutils.object_detection import non_max_suppression
import numpy as np 
import cv2 
import imutils

image = cv2.imread("6.jpg")
scale = 1.0
w = int(image.shape[1] / scale)
image = imutils.resize(image, width=min(400, image.shape[1]))

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(rects,_) = hog.detectMultiScale(image, winStride=(4, 4),
                                        padding=(8, 8), scale=1.05)

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

for (xA, yA, xB, yB) in pick:
             cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)


cv2.imshow("Deteccion", image)
cv2.waitKey(0)
cv2.destroyAllWindows()