# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:39:27 2021

@author: chomi
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.keras
from PIL import Image, ImageOps
from detecto import core
import numpy as np
import cv2 as cv

np.set_printoptions(suppress=True)
np.set_printoptions(precision =3)

#importing tranined moddels
model = tensorflow.keras.models.load_model('new_keras.h5', compile=False)
cdmodel = core.Model.load('model.pth', ['sample'])

def find(string):
    path = 'C:/Users/chomi/Desktop/ten/z'
    for i in range(10):
        if(string in os.listdir(path + str(i))):
            return i
k = []
#taking prediction and classifying volume
def classify(prediction):
    for i in range(10):
        if(prediction[0][i] > 0.7):
            print(' volume is', i)
            return
    idx = np.where(prediction[0] == np.amax(prediction[0]))
    prediction[0][idx[0][0]] += 0.8
    print(" Unclear but most likely", end='')
    classify(prediction)
    
def evaluate():
    path = 'C:/Users/chomi/Desktop/all_small/'
    counter = 0
    for filename in os.listdir(path):
        if(not filename.endswith('.png')):
            continue
        counter += 1
        print(counter, end=' :: '+filename)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        #crucible detection
        img = Image.open(path + filename)
        pred = cdmodel.predict(img)
        lbl, box, score = pred
        temp = box.numpy()[0]
        box = (round(temp[0]), round(temp[1]), round(temp[2]), round(temp[3]))
        # print(filename, find(filename))
        image = img.crop(box) #passing image to powder model
        
        # formatting image for powder stuff
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        
        #display image (to user) that model took as input
        # cvim = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        # cvim = cv.resize(cvim, size, interpolation = cv.INTER_AREA)
        # cv.imshow('sample', cvim)
        # cv.waitKey(1)
        
        #run image through the model
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        global k 
        k.append(prediction)
        classify(prediction) 

    cv.destroyAllWindows()
