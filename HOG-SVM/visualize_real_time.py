import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
import joblib,cv2
from skimage import color
import Sliding as sd
size = (64,128)
step_size = (10,10)
downscale = 1.25
from tkinter import filedialog
model = joblib.load('models/models.dat')
archivo_abierto=filedialog.askopenfilename(initialdir = "/",
                title = "Seleccione archivo",filetypes = (("mp4 files","*.mp4"),
                ("all files","*.*")))
cap = cv2.VideoCapture(archivo_abierto) 
while True:
    ret,frame = cap.read()
    image = cv2.resize()
    detections = []
    scale = 0
    for im_scaled in pyramid_gaussian(image, downscale = downscale):
     
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break
        for (x, y, window) in sd.sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue
            window = color.rgb2gray(window)
            fd=hog(window, orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
            fd = fd.reshape(1, -1)
            pred = model.predict(fd)
            if pred == 1:
                if model.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fd), 
                    int(size[0] * (downscale**scale)),
                    int(size[1] * (downscale**scale))))
        scale += 1
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

    for(x1, y1, x2, y2) in pick:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Deteccion',image)
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()

                
    


