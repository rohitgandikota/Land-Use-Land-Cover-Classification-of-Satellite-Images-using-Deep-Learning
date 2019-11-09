# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:07:31 2019

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
from keras.regularizers import l2
from keras.utils import plot_model
from ann_visualizer.visualize import ann_viz

num_classes = 4
data_path = 'C:\\Users\\user\\Desktop\\Projects\\FeatureExtractionSatelliteImagery\\SatellitePixelLearning-master\\processing\\Data\\Train'

def listdir_full(d):
    return [os.path.join(d,f) for f in os.listdir(d)]
    
water_path = listdir_full(data_path+'\\water')
cloud_path = listdir_full(data_path+'\\cloud')
veg_path = listdir_full(data_path+'\\veg')
urban_path = listdir_full(data_path+'\\urban')
land_path = listdir_full(data_path+'\\land')
snow_path = listdir_full(data_path+'\\snow')


data_path = [cloud_path, water_path, veg_path, land_path, urban_path,  snow_path]
i = 0
for datum_path in data_path:
    class_data = np.zeros((4,1))
    for path in datum_path:
        dataset = gdal.Open(path)
        data = dataset.ReadAsArray()
    #    args = np.argsnonzero(data[0])
        data_new = data[:,data[0]!= -99]
        data_new = data_new[:,data_new[0]!= 0]
        class_data = np.hstack((class_data, data_new))
    class_data = class_data[:,1:]
    ind_list = [i for i in range(np.shape(class_data)[1])]
    shuffle(ind_list)
    class_data = class_data[:, ind_list]
    class_data = class_data[:,:min(60000,np.shape(class_data)[1])]
    if i == 0:
        X = class_data
        Y = np.zeros((np.shape(class_data)[1], 1))
    else:
        X = np.hstack((X,class_data))
        labels = np.full((np.shape(class_data)[1], 1),i)
        Y = np.vstack((Y,labels))
    i = i+1
    print(path + ': ' +str(np.shape(class_data)[1]))

Y[Y==5] = 4
Y[Y==4] = 3
#Y[Y==3] = 2
X = np.swapaxes(X,0,1)
Y = utils.to_categorical(Y, num_classes)

#WI = np.divide((X[:,2] - X[:,3]), (X[:,2] + X[:,3]))
#WI = (WI - min(WI)) / (max(WI) - min(WI))
#
#
#VI = np.divide((X[:,2] - X[:,0]), (X[:,2] + X[:,0]))
#VI = (VI - min(VI)) / (max(VI) - min(VI))
#
#VI = np.reshape(VI, (len(VI),1))
#WI = np.reshape(WI, (len(WI),1))
#X = np.hstack((X,WI,VI))

 
ind_list = [i for i in range(len(X))]
shuffle(ind_list)
X  = X[ind_list, :]
Y = Y[ind_list, :]

#X_test = X[:100000]
#Y_test = Y[:100000]
#
#X = X[100000:]
#Y = Y[100000:]
#Model3
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(4,)))
model.add(BatchNormalization())
model.add(Dense(32,  activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,  activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32,  activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax', name='Output'))

#model = Sequential()
#model.add(Dense(16, activation='relu',  input_shape=(6,)))
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

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(),
              metrics=['accuracy'])

history = model.fit(X, Y,
                    batch_size = 250000,
                    epochs= 400,
                    verbose=1,
                    validation_split=0.1)

#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

model.save("C:\\Users\\user\\Desktop\\Projects\\FeatureExtractionSatelliteImagery\\SatellitePixelLearning-master\\processing\\Models\\Model3.h5")
print("Saved model to disk")

plot_model(model, to_file="C:\\Users\\user\\Desktop\\Projects\\FeatureExtractionSatelliteImagery\\SatellitePixelLearning-master\\processing\\Models\\BestModel.png")


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
