# *_*coding:utf-8 *_*

import os
import sys
import cv2
import logging
import numpy as np

def logger_init():
    logger = logging.getLogger("PedestranDetect")
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter 
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def load_data_set(logger):
  
    logger.info('Comprobando la ruta de datos!')
    pwd = os.getcwd()
    logger.info('Ruta informacion:{}'.format(pwd))

    # Extraer muestras positivas
    pos_dir = os.path.join(pwd, 'Positive')
    if os.path.exists(pos_dir):
        logger.info('Positivo ruta:{}'.format(pos_dir))
        pos = os.listdir(pos_dir)
        logger.info('Positivo cantidad:{}'.format(len(pos)))

    # Extraer muestras negativas
    neg_dir = os.path.join(pwd, 'Negative')
    if os.path.exists(neg_dir):
        logger.info('Negativo ruta:{}'.format(neg_dir))
        neg = os.listdir(neg_dir)
        logger.info('Negativo cantidad:{}'.format(len(neg)))

    # Extraer conjunto de prueba
    test_dir = os.path.join(pwd, 'TestData')
    if os.path.exists(test_dir):
        logger.info('Test ruta:{}'.format(test_dir))
        test = os.listdir(test_dir)
        logger.info('Test cantidad:{}'.format(len(test)))

    return pos, neg, test

def load_train_samples(pos, neg):
    '''
         Combine la posición de muestra positiva y la posición de muestra negativa para crear un conjunto de datos de entrenamiento y el conjunto de etiquetas correspondiente
         : param pos: lista de nombres de archivo de muestra positivos
         : param neg: lista de nombres de archivo de muestra negativos
         : muestras de retorno: lista de nombres de archivos de muestra de entrenamiento combinados
         : etiquetas de retorno: una lista de etiquetas correspondientes a las muestras de entrenamiento
    '''
    pwd = os.getcwd()
    pos_dir = os.path.join(pwd, 'Positive')
    neg_dir = os.path.join(pwd, 'Negative')

    samples = []
    labels = []
    for f in pos:
        file_path = os.path.join(pos_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(1.)

    for f in neg:
        file_path = os.path.join(neg_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(-1.)

    # etiquetas que se convertirán en una matriz numpy
    labels = np.int32(labels)
    labels_len = len(pos) + len(neg)
    labels = np.resize(labels, (labels_len, 1))

    return samples, labels

def extract_hog(samples, logger):
    '''
         Extraiga las características de HOG del conjunto de datos de entrenamiento y regrese
         : muestras de parámetros: conjunto de datos de entrenamiento
         : param logger: módulo de impresión de información de registro
         : tren de retorno: características HOG extraídas del conjunto de datos de entrenamiento
    '''
    train = []
    logger.info('Extracting HOG Descriptors...')
    num = 0.
    total = len(samples)
    for f in samples:
        num += 1.
        logger.info('Processing {} {:2.1f}%'.format(f, num/total*100))
        hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
        img = cv2.imread(f, -1)
        img = cv2.resize(img, (64,128))
        descriptors = hog.compute(img)
        logger.info('hog tamaño del descriptor: {}'.format(descriptors.shape))    
        train.append(descriptors)

    train = np.float32(train)
    train = np.resize(train, (total, 3780))

    return train

def get_svm_detector(svm):
    '''
         Exporte el detector SVM que puede usarse en cv2.HOGDescriptor (), que es esencialmente una lista de vectores de soporte SVM entrenados y parámetros rho
         : param svm: clasificador SVM entrenado
    : return: Una lista de vectores de soporte SVM y parámetros rho, que se pueden usar como un detector SVM para cv2.HOGDescriptor ()
    '''
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)

def train_svm(train, labels, logger):
    '''
         Clasificador de SVM de tren
         : tren param: conjunto de datos de entrenamiento
         : etiquetas param: las etiquetas correspondientes al conjunto de entrenamiento
         : param logger: módulo de impresión de información de registro
         : return: detector SVM (nota: el svm en hogdescriptor de opencv no puede usar directamente el modelo svm de opencv, sino exportar la matriz de formato correspondiente)
    '''
    logger.info('Configuring SVM classifier.')
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1) 
    svm.setC(0.01) 
    svm.setType(cv2.ml.SVM_EPS_SVR)

    logger.info('Inicio de entrenamiento')
    svm.train(train, cv2.ml.ROW_SAMPLE, labels)
    logger.info('Entrenamiento realizado')
    pwd = os.getcwd()
    model_path = os.path.join(pwd, 'svm.xml')
    svm.save(model_path)
    logger.info('El clasificador SVM entrenado: {}'.format(model_path))

    return get_svm_detector(svm)

def test_hog_detect(test, svm_detector, logger):
    '''
         Importar conjunto de prueba, resultados de prueba
         : prueba param: conjunto de datos de prueba
         : param svm_detector: detector SVM para HOGDescriptor
         : param logger: módulo de impresión de información de registro
         : volver: ninguno
    '''
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(svm_detector)
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    pwd = os.getcwd()
    test_dir = os.path.join(pwd, 'TestData')
    cv2.namedWindow('Detector')
    for f in test:
        file_path = os.path.join(test_dir, f)
        logger.info('Procesando {}'.format(file_path))
        img = cv2.imread(file_path)
        rects, _ = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
        for (x,y,w,h) in rects:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.imshow('Detector', img)
        c = cv2.waitKey(0) & 0xff
        if c == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logger = logger_init()
    pos, neg, test = load_data_set(logger=logger)
    samples, labels = load_train_samples(pos, neg)
    train = extract_hog(samples, logger=logger)
    logger.info('Tamaño de muestra: {}'.format(train.shape))
    logger.info('Tamaño de etiquetas: {}'.format(labels.shape))
    svm_detector = train_svm(train, labels, logger=logger)
    test_hog_detect(test, svm_detector, logger)