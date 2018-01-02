import keras
import numpy as np
import os
import cv2

from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Activation, Dropout, BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate

class batchLoader():
    
    def __init__(self):
    
        self.filesDir="datafiles/moves_cnn_data/"

        self.filenames = [ file for file in os.listdir(self.filesDir)]
        self.filenameIdx=0
        self.batchLen=10

        self.dataFile=np.load(self.filesDir+self.filenames[self.filenameIdx])
        self.fileIdx=0


    def getBatch(self):

        imgs=[]
        res=[]
        
        currFileIdx=self.fileIdx*self.batchLen
        
        for i in range(currFileIdx,currFileIdx+self.batchLen):
            imgs.append(self.dataFile[:,0][i])
            res.append(self.dataFile[:,1][i])

        self.fileIdx+=1
        
        if (self.fileIdx*self.batchLen)>=len(self.dataFile)-self.batchLen:
            self.fileIdx=0
            self.filenameIdx+=1
            if self.filenameIdx>=len(self.filenames):
                return np.array(imgs),np.array(deck),False
            self.dataFile=np.load(self.filesDir+self.filenames[self.filenameIdx])

        return np.array(imgs),np.array(deck),True

batches=batchLoader()


input_shape=(460, 330, 3)
    
board=Input(shape=input_shape,name="board-img")

model=Convolution2D(64,kernel_size=(3,3),strides=(1,1))(board)
model=BatchNormalization()(model)
model=Activation('relu')(model)

model=Convolution2D(128,kernel_size=(5,5),strides=(1,1))(model)
model=BatchNormalization()(model)
model=Activation('relu')(model)

model=MaxPooling2D()(model)

model=Convolution2D(128,kernel_size=(3,3),strides=(1,1))(model)
model=BatchNormalization()(model)
model=Activation('relu')(model)

model=MaxPooling2D()(model)

model=Flatten()(model)

model=Dense(128)(model)
model=Activation('relu')(model)
model=Dropout(0.2)(model)

model=Dense(1)(model)
output=Activation('sigmoid')(model)

cardPredictModel=Model(board,output)
cardPredictModel.summary()
cardPredictModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

dataAvailable=True

while dataAvailable:
    print("Fitting data - file number ",batches.filenameIdx+1,"/",len(batches.filenames)," with images ",batches.fileIdx*batches.batchLen," to ",(batches.fileIdx*batches.batchLen)+batches.batchLen," out of ",len(batches.dataFile))
    x1,y,dataAvailable=batches.getBatch()
    cardPredictModel.fit(x1,y,verbose=1)
    
cardPredictModel.save('moves_predictor.h5')