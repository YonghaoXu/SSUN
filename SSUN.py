# -*- coding: utf-8 -*-
"""
@author: Yonghao.Xu
"""


import scipy.io as sio  
import numpy as np  
import matplotlib.pyplot as plt
from HyperFunctions import*

from keras.datasets import mnist
from keras.models import Model,Sequential,save_model,load_model
from keras.layers import Input, Dense, Activation,LSTM,merge,Convolution2D, MaxPooling2D,AveragePooling2D, Flatten,Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import np_utils



def LSTMMCNN_RS(time_step,nb_features,num_PC,img_rows,img_cols):
    LSTMInput = Input(shape=(time_step,nb_features),name='LSTMInput')

    LSTMSpectral = LSTM(128,name='LSTMSpectral',consume_less='gpu',W_regularizer=l2(0.0001),U_regularizer=l2(0.0001))(LSTMInput)
    
    LSTMDense = Dense(128,activation='relu', name='LSTMDense')(LSTMSpectral)
   
    LSTMSOFTMAX = Dense(nb_classes,activation='softmax', name='LSTMSOFTMAX')(LSTMDense)    
    
    CNNInput = Input(shape=[num_PC,img_rows,img_cols],name='CNNInput')

    CONV1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='CONV1')(CNNInput) 
    POOL1 = MaxPooling2D((1, 2, 2), strides=(2,2), name='POOL1')(CONV1)   
    CONV2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='CONV2')(POOL1)
    POOL2 = MaxPooling2D((1,2, 2), strides=(2,2), name='POOL2')(CONV2)
    CONV3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='CONV3')(POOL2)
    POOL3 = MaxPooling2D((1,2,2), strides=(2,2), name='POOL3')(CONV3)
    
    FLATTEN1 = Flatten(name='FLATTEN1')(POOL1)
    FLATTEN2 = Flatten(name='FLATTEN2')(POOL2)
    FLATTEN3 = Flatten(name='FLATTEN3')(POOL3)
    
    DENSE1 = Dense(128,activation='relu', name='DENSE1')(FLATTEN1)
    DENSE2 = Dense(128,activation='relu', name='DENSE2')(FLATTEN2)
    DENSE3 = Dense(128,activation='relu', name='DENSE3')(FLATTEN3)
    
    CNNDense = merge([DENSE1, DENSE2, DENSE3], mode='sum', name='CNNDense')

    
    CNNSOFTMAX = Dense(nb_classes,activation='softmax', name='CNNSOFTMAX')(CNNDense)
    

    JOINT = merge([LSTMDense,CNNDense],mode='concat')
    JOINTDENSE = Dense(128,activation='relu', name='JOINTDENSE')(JOINT)

    JOINTSOFTMAX = Dense(nb_classes,activation='softmax',name='JOINTSOFTMAX')(JOINTDENSE)


    model = Model(input=[LSTMInput,CNNInput], output=[JOINTSOFTMAX,LSTMSOFTMAX,CNNSOFTMAX])    
    
    
    rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=1e-05)
    
    model.compile(optimizer=rmsp, loss='categorical_crossentropy',
                  metrics=['accuracy'],loss_weights=[1, 1,1])

    return model
    
    
def LSTM_RS(time_step,nb_features):
    LSTMInput = Input(shape=(time_step,nb_features),name='LSTMInput')

    LSTMSpectral = LSTM(128,name='LSTMSpectral',consume_less='gpu',W_regularizer=l2(0.0001),U_regularizer=l2(0.0001))(LSTMInput)
    
    LSTMDense = Dense(128,activation='relu', name='LSTMDense')(LSTMSpectral)
    LSTMSOFTMAX = Dense(nb_classes,activation='softmax', name='LSTMSOFTMAX')(LSTMDense)
    
    model = Model(input=[LSTMInput], output=[LSTMSOFTMAX])
    rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=1e-05)
    
    model.compile(optimizer=rmsp, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    
    
def MCNN_RS(num_PC,img_rows,img_cols):
    CNNInput = Input(shape=[num_PC,img_rows,img_cols],name='CNNInput')

    CONV1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='CONV1')(CNNInput)
    POOL1 = MaxPooling2D((1, 2, 2), strides=(2,2), name='POOL1')(CONV1)
    CONV2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='CONV2')(POOL1)
    POOL2 = MaxPooling2D((1,2, 2), strides=(2,2), name='POOL2')(CONV2)
    CONV3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='CONV3')(POOL2)
    POOL3 = MaxPooling2D((1,2,2), strides=(2,2), name='POOL3')(CONV3)
    
    FLATTEN1 = Flatten(name='FLATTEN1')(POOL1)
    FLATTEN2 = Flatten(name='FLATTEN2')(POOL2)
    FLATTEN3 = Flatten(name='FLATTEN3')(POOL3)
    
    DENSE1 = Dense(128,activation='relu', name='DENSE1')(FLATTEN1)
    DENSE2 = Dense(128,activation='relu', name='DENSE2')(FLATTEN2)
    DENSE3 = Dense(128,activation='relu', name='DENSE3')(FLATTEN3)
    
    CNNDense = merge([DENSE1, DENSE2, DENSE3], mode='sum', name='CNNDense')

    
    CNNSOFTMAX = Dense(nb_classes,activation='softmax', name='CNNSOFTMAX')(CNNDense)
    
    
    model = Model(input=[CNNInput], output=[CNNSOFTMAX])
    rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=1e-05)
    
    model.compile(optimizer=rmsp, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
   

    
   
#%% Spectral

w=2
num_PC=1
israndom=True
randtime = 1
OASpectral_Pavia = np.zeros((9+2,randtime))
s1s2=2
time_step = 3

    
for r in range(0,randtime):

    #################Pavia#################
    dataID=1
    data = HyperspectralSamples(dataID=dataID, timestep=time_step, w=w, num_PC=num_PC, israndom=israndom, s1s2=s1s2)
    X = data[0]
    X_train = data[1]
    X_test = data[2]
    XP = data[3]
    XP_train = data[4]
    XP_test = data[5]
    Y = data[6]-1
    Y_train = data[7]-1
    Y_test = data[8]-1
    
    batch_size = 128
    
    nb_classes = Y_train.max()+1
    nb_epoch = 500
    nb_features = X.shape[-1]
    
    img_rows, img_cols = XP.shape[-1],XP.shape[-1]
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(Y_train, nb_classes)
    y_test = np_utils.to_categorical(Y_test, nb_classes)
    
    model = LSTM_RS(time_step=time_step,nb_features=nb_features)
    
    histloss=model.fit([X_train], [y_train], nb_epoch=nb_epoch, batch_size=batch_size, verbose=0, shuffle=True)
    losses = histloss.history
    

    PredictLabel = model.predict([X_test],verbose=0).argmax(axis=-1)

    OA,Kappa,ProducerA = CalAccuracy(PredictLabel,Y_test[:,0])    
    OASpectral_Pavia[0:9,r] = ProducerA
    OASpectral_Pavia[-2,r] = OA
    OASpectral_Pavia[-1,r] = Kappa
    print('rand',r+1,'LSTM Pavia test accuracy:', OA*100)

    Map = model.predict([X],verbose=0)

    Spectral = Map.argmax(axis=-1)

    X_result = DrawResult(Spectral,1)

    plt.imsave('LSTM_Pavia_r'+repr(r+1)+'OA_'+repr(int(OA*10000))+'.png',X_result)
      
   
      

# Spatial
w=28
num_PC=4
israndom=True
randtime = 1
OASpatial_Pavia = np.zeros((9+2,randtime))
s1s2=1
time_step=1

    
for r in range(0,randtime):

    #################Pavia#################
    dataID=1
    data = HyperspectralSamples(dataID=dataID, timestep=time_step, w=w, num_PC=num_PC, israndom=israndom, s1s2=s1s2)
    X = data[0]
    X_train = data[1]
    X_test = data[2]
    XP = data[3]
    XP_train = data[4]
    XP_test = data[5]
    Y = data[6]-1
    Y_train = data[7]-1
    Y_test = data[8]-1
    
    batch_size = 128
    
    nb_classes = Y_train.max()+1
    nb_epoch = 500
    nb_features = X.shape[-1]
    
    img_rows, img_cols = XP.shape[-1],XP.shape[-1]
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(Y_train, nb_classes)
    y_test = np_utils.to_categorical(Y_test, nb_classes)
    
    model = MCNN_RS(num_PC,img_rows,img_cols)
    
    histloss=model.fit([XP_train], [y_train], nb_epoch=nb_epoch, batch_size=batch_size, verbose=0, shuffle=True)
    losses = histloss.history
    

    PredictLabel = model.predict([XP_test],verbose=0).argmax(axis=-1)

    OA,Kappa,ProducerA = CalAccuracy(PredictLabel,Y_test[:,0])    
    OASpatial_Pavia[0:9,r] = ProducerA
    OASpatial_Pavia[-2,r] = OA
    OASpatial_Pavia[-1,r] = Kappa
    print('rand',r+1,'MCNN Pavia test accuracy:', OA*100)

    Map = model.predict([XP],verbose=0)

    Spatial = Map.argmax(axis=-1)

    X_result = DrawResult(Spatial,1)

    plt.imsave('MCNN_Pavia_r'+repr(r+1)+'OA_'+repr(int(OA*10000))+'.png',X_result)
      
   

# Joint
time_step=3
w=28
num_PC=4
israndom=True
s1s2=2
randtime = 1
OAJoint_Pavia = np.zeros((9+2,randtime))

    
for r in range(0,randtime):

    #################Pavia#################
    dataID=1
    data = HyperspectralSamples(dataID=dataID, timestep=time_step, w=w, num_PC=num_PC, israndom=israndom, s1s2=s1s2)
    X = data[0]
    X_train = data[1]
    X_test = data[2]
    XP = data[3]
    XP_train = data[4]
    XP_test = data[5]
    Y = data[6]-1
    Y_train = data[7]-1
    Y_test = data[8]-1
    
    batch_size = 128
    
    nb_classes = Y_train.max()+1
    nb_epoch = 500
    nb_features = X.shape[-1]
    
    img_rows, img_cols = XP.shape[-1],XP.shape[-1]
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(Y_train, nb_classes)
    y_test = np_utils.to_categorical(Y_test, nb_classes)
    
    
    model = LSTMMCNN_RS(time_step=time_step,nb_features=nb_features,num_PC=num_PC,img_rows=img_rows,img_cols=img_cols)
    histloss=model.fit([X_train,XP_train], [y_train,y_train,y_train], nb_epoch=nb_epoch, batch_size=batch_size, verbose=0, shuffle=True)
    losses = histloss.history
    

    PredictLabel = model.predict([X_test,XP_test],verbose=0)[0].argmax(axis=-1)

    OA,Kappa,ProducerA = CalAccuracy(PredictLabel,Y_test[:,0])    
    OAJoint_Pavia[0:9,r] = ProducerA
    OAJoint_Pavia[-2,r] = OA
    OAJoint_Pavia[-1,r] = Kappa
    print('rand',r+1,'LSTMMCNN Pavia test accuracy:', OA*100)

    Map = model.predict([X,XP],verbose=0)

    Joint = Map[0].argmax(axis=-1)

    X_result = DrawResult(Joint,1)

    plt.imsave('LSTMMCNN_Pavia_r'+repr(r+1)+'OA_'+repr(int(OA*10000))+'.png',X_result)
      
    
     
save_fn = 'OAAll.mat'
sio.savemat(save_fn, {'OASpectral_Pavia': OASpectral_Pavia,'OASpatial_Pavia': OASpatial_Pavia,'OAJoint_Pavia': OAJoint_Pavia})


