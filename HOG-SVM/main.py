from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import datetime
import cv2,joblib   
import imutils
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox
import os
from PIL import ImageTk, Image
from skimage.feature import hog
import joblib,glob,os,cv2
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, metrics
import numpy as np 
from sklearn.preprocessing import LabelEncoder

class Ventana:
    def __init__(self, master):
        self.master = master
        master.title("HOG + SVM")
        self.img = PhotoImage(file="Logos/espe.png")
        self.widget = Label(master, image=self.img).pack()
 
        self.proyecto_label= Label(master, text="Proyecto de Inteligencia Artificial")
        self.proyecto_label.config(padx=10, pady=10,font=('Arial', 14))
        self.proyecto_label.pack()
        self.tema_label= Label(master, text="Reconocimiento de Peatones utilizando \n HOG + SVM")
        self.tema_label.config(padx=10, pady=10,font=('Arial', 14))
        self.tema_label.pack()
        self.integrantes_label= Label(master, text="Patricia Sambachi \nPatricio Vinueza")
        self.integrantes_label.config(padx=10, pady=10,font=('Arial', 14))
        self.integrantes_label.pack()
        
        self.opciones_label= Label(master, text="Opciones")
        self.opciones_label.config(padx=10, pady=10,font=('Arial', 14))
        self.opciones_label.pack()
        
        self.botonImagen = Button(master, text="Deteccion en imagenes", command=self.imagenD,relief=SOLID)
        self.botonImagen.config(padx=10, pady=10,font=('Arial', 12))
        self.botonImagen.pack()
        self.opciones_label= Label(master, text="")
        self.opciones_label.config(padx=5, pady=5,font=('Arial', 1))
        self.opciones_label.pack()

        self.botonVideo = Button(master, text="Deteccion en videos", command=self.videoD,relief=SOLID)
        self.botonVideo.config(padx=10, pady=10,font=('Arial', 12))
        self.botonVideo.pack()

        self.opciones_label= Label(master, text="")
        self.opciones_label.config(padx=5, pady=5,font=('Arial', 1))
        self.opciones_label.pack()

        self.botonEntrenar = Button(master, text="Entrenar Modelo", command=self.train,relief=SOLID)
        self.botonEntrenar.config(padx=10, pady=10,font=('Arial', 12))
        self.botonEntrenar.pack()

    def imagenD(self):
        fileName = r"C:/Users/pato/Desktop/Proyecto de IA/models/models.dat"
        os.path.exists(fileName)
        if os.path.exists(fileName):
            archivoV=filedialog.askopenfilename(initialdir = "/",
                    title = "Seleccione archivo",filetypes = (("jpeg files","*.jpg"),
                    ("all files","*.*")))
            font = cv2.FONT_HERSHEY_SIMPLEX
            image = cv2.imread(archivoV)
            scale = 1.0
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            orig = image.copy()
            start = datetime.datetime.now()
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                    padding=(8, 8), scale=1.05)
            a =  len(weights)
            idx = 0
            for (x, y, w, h) in rects:
                    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    confidence = float((weights[idx]))
                    cv2.putText(orig,"{:.5f}".format(confidence), (x, y+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    idx += 1
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

            cv2.imshow("Deteccion", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else :
            messagebox.showinfo("Information","No existe modelo entrenado")

    def videoD(self):
        fileName = r"C:/Users/pato/Desktop/Proyecto de IA/models/models.dat"
        os.path.exists(fileName)
        if os.path.exists(fileName):

            archivo_abierto=filedialog.askopenfilename(initialdir = "/",
                    title = "Seleccione archivo",filetypes = (("mp4 files","*.mp4"),
                    ("all files","*.*")))
            hog = cv2.HOGDescriptor() 
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
            cap = cv2.VideoCapture(archivo_abierto) 
            while cap.isOpened(): 
                ret, image = cap.read() 
                if ret: 
                    image = imutils.resize(image,  
                                        width=min(400, image.shape[1])) 
                    (regions, _) = hog.detectMultiScale(image, 
                                                        winStride=(4, 4), 
                                                        padding=(4, 4), 
                                                        scale=1.05) 
                    for (x, y, w, h) in regions: 
                        cv2.rectangle(image, (x, y), 
                                    (x + w, y + h),  
                                    (0, 255, 0), 2) 
                    cv2.imshow("Deteccion", image) 
                    if cv2.waitKey(25) & 0xFF == ord('q'): 
                        break
                else: 
                    break
            cap.release()
            cv2.destroyAllWindows()
        else :
            messagebox.showinfo("Information","No existe modelo entrenado")

    def train(self):
        messagebox.showinfo("Information","Entrenando modelo")
        train_data = []
        train_labels = []
        pos_im_path = 'dataset/positive/'
        neg_im_path = 'dataset/negative/'
        model_path = 'models/models.dat'

        for filename in glob.glob(os.path.join(pos_im_path,"*.png")):
            fd = cv2.imread(filename,0)
            fd = cv2.resize(fd,(64,128))
            fd = hog(fd,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
            train_data.append(fd)
            train_labels.append(1)
        for filename in glob.glob(os.path.join(neg_im_path,"*.png")):
            fd = cv2.imread(filename,0)
            fd = cv2.resize(fd,(64,128))
            fd = hog(fd,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
            train_data.append(fd)
            train_labels.append(0)
        train_data = np.float32(train_data)
        train_labels = np.array(train_labels)
        print('Preparando datos')
        print('Train Data:',len(train_data))
        print('Train Labels (1,0)',len(train_labels))
        print("Clasificador SVM")

        model = LinearSVC()
        print('Entrenando...... SVM')
        model.fit(train_data,train_labels)
        joblib.dump(model, 'models/models.dat')
        print('Modelo guardado : {}'.format('models/models.dat'))
        messagebox.showinfo("Information","Modelo entrenado")
        
root = Tk()
root.geometry("600x560")
miVentana = Ventana(root)
root.mainloop()