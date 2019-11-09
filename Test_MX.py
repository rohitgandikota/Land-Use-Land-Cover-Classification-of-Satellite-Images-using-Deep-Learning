# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:01:31 2019

@author: Rohit Gandikota (NR02440)
"""

from osgeo import gdal, ogr, osr
from keras import *
import os 

from random import shuffle
import numpy as np
import math
from keras.optimizers import *
from keras.layers import *
import matplotlib.pyplot as plt

# Making a test image
test_path = 'C:\\Users\\user\\Desktop\\Projects\\FeatureExtractionSatelliteImagery\\SatellitePixelLearning-master\\processing\\Test_files\\Mar'
datum = os.listdir(test_path)
i = 0
for data in datum:
    print(data)
    test_img_path = test_path + '\\' + data
    t_dataset = gdal.Open(test_img_path)
    band = t_dataset.ReadAsArray()
    print(np.shape(band))
    
    i+=1
    if i == 1:
        test_image = np.stack((band,band))
    else:
        band = np.expand_dims(band, axis=0)
        test_image = np.vstack((test_image, band)) 
test_image = test_image[1:]

X_test = np.reshape(test_image, (4,-1))
X_test = np.swapaxes(X_test,0,1)

X_test_latent = encoder.predict(X_test,batch_size=1000000, verbose=1)

#WI = np.divide((X_test[:,2] - X_test[:,3]), (X_test[:,2] + X_test[:,3]))
#WI[WI!= WI] = min(WI[WI==WI])
#WI = (WI - min(WI)) / (max(WI) - min(WI))
#
#
#VI = np.divide((X_test[:,2] - X_test[:,0]), (X_test[:,2] + X_test[:,0]))
#VI[VI!= VI] = min(VI[VI==VI])
#VI = (VI - min(VI)) / (max(VI) - min(VI))
#
#VI = np.reshape(VI, (len(VI),1))
#WI = np.reshape(WI, (len(WI),1))
#X_test = np.hstack((X_test,WI,VI))


#model = Sequential()
#model.add(Dense(16, activation='relu',  input_shape=(4,)))
#model.add(BatchNormalization())
#model.add(Dense(32,  activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(64,  activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(32,  activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(16, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(num_classes, activation='softmax'))
#
#model.load_weights("C:\\Users\\user\\Desktop\\Projects\\FeatureExtractionSatelliteImagery\\SatellitePixelLearning-master\\processing\\Final Codes\\BestModel.h5")

labels_test = model.predict_classes(X_test_latent,batch_size=1000000, verbose=1)
labels_test = labels_test.reshape((t_dataset.RasterYSize, t_dataset.RasterXSize))

def WriteRaster(InputArray, OutputFile, NROWS, NCOLS, wkt_projection, geotransform):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(OutputFile, NCOLS, NROWS, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(InputArray)
    dataset.FlushCache()
    return None

WriteRaster(labels_test, 'C:\\Users\\user\\Desktop\\Projects\\FeatureExtractionSatelliteImagery\\SatellitePixelLearning-master\\processing\\Test_Results\\Autoenc_Model3_Mar_New.tif', t_dataset.RasterYSize, t_dataset.RasterXSize,t_dataset.GetProjection(),t_dataset.GetGeoTransform())



cloud_percentage = (len(labels_test[labels_test==0])/len(labels_test[labels_test==labels_test]))*100
