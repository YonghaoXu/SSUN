# -*- coding: utf-8 -*-
"""
@author: Yonghao.Xu
"""

# SSUN Related Functions

import scipy.io as sio  
import numpy as np  
import matplotlib.pyplot as plt

def featureNormalize(X,type):
    #type==1 x = (x-mean)/std(x)
    #type==2 x = (x-max(x))/(max(x)-min(x))
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
        return X_norm    
    
def PCANorm(X,num_PC):
    mu = np.mean(X,0)
    X_norm = X-mu
    
    Sigma = np.cov(X_norm.T)
    [U, S, V] = np.linalg.svd(Sigma)   
    XPCANorm = np.dot(X_norm,U[:,0:num_PC])
    return XPCANorm
    
def MirrowCut(X,hw):
    #X  size: row * column * num_feature

    [row,col,n_feature] = X.shape

    X_extension = np.zeros((3*row,3*col,n_feature))
    
    for i in range(0,n_feature):
        lr = np.fliplr(X[:,:,i])
        ud = np.flipud(X[:,:,i])
        lrud = np.fliplr(ud)
        
        l1 = np.concatenate((lrud,ud,lrud),axis=1)
        l2 = np.concatenate((lr,X[:,:,i],lr),axis=1)
        l3 = np.concatenate((lrud,ud,lrud),axis=1)
        
        X_extension[:,:,i] = np.concatenate((l1,l2,l3),axis=0)
    
    X_extension = X_extension[row-hw:2*row+hw,col-hw:2*col+hw,:]
    
    return X_extension
    
def DrawResult(labels,imageID):
    #ID=1:Pavia University
    #ID=2:Indian Pines
    #ID=6:KSC
    num_class = labels.max()+1
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    elif imageID == 2:
        row = 145
        col = 145
        palette = np.array([[255,0,0],
                            [0,255,0],
                            [0,0,255],
                            [255,255,0],
                            [0,255,255],
                            [255,0,255],
                            [176,48,96],
                            [46,139,87],
                            [160,32,240],
                            [255,127,80],
                            [127,255,212],
                            [218,112,214],
                            [160,82,45],
                            [127,255,0],
                            [216,191,216],
                            [238,0,0]])
        palette = palette*1.0/255
    elif imageID == 6:
        row = 512
        col = 614
        palette = np.array([[94, 203, 55],
                            [255, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [255, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255]])
        palette = palette*1.0/255
    
    X_result = np.zeros((labels.shape[0],3))
    for i in range(0,num_class):
        X_result[np.where(labels==i),0] = palette[i,0]
        X_result[np.where(labels==i),1] = palette[i,1]
        X_result[np.where(labels==i),2] = palette[i,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))
    
    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    return OA,Kappa,producerA
   
def HyperspectralSamples(dataID=1, timestep=4, w=24, num_PC=3, israndom=False, s1s2=2):   
    #dataID=1:Pavia University
    #dataID=2:Indian
    #dataID=6:KSC

    if dataID==1:
        data = sio.loadmat('/data/yonghao.xu/HSI/DataSets/PaviaU.mat')
        X = data['paviaU']
    
        data = sio.loadmat('/data/yonghao.xu/HSI/DataSets/PaviaU_gt.mat')
        Y = data['paviaU_gt']
        
        train_num_array = [548, 540, 392, 542, 256, 532, 375, 514, 231]
    elif dataID==2:
        data = sio.loadmat('/data/yonghao.xu/HSI/DataSets/Indian_pines_corrected.mat')
        X = data['data']
    
        data = sio.loadmat('/data/yonghao.xu/HSI/DataSets/Indian_pines_gt.mat')
        Y = data['groundT']
        
        train_num_array = [30, 150, 150, 100, 150, 150, 20, 150, 15, 150, 150, 150, 150, 150, 50, 50]
    elif dataID==6:
        data = sio.loadmat('/data/yonghao.xu/HSI/DataSets/KSC.mat')
        X = data['KSC']
        X[np.where(X>700)] = 0
        data = sio.loadmat('/data/yonghao.xu/HSI/DataSets/KSC_gt.mat')
        Y = data['KSC_gt']
        
        train_num_array = [33, 23, 24, 24, 15, 22, 9, 38, 51, 39, 41, 49, 91]
   
    train_num_array = np.array(train_num_array).astype('int')
    [row,col,n_feature] = X.shape
    K = row*col
    X = X.reshape(row*col, n_feature)    
    Y = Y.reshape(row*col, 1)    
    
    n_class = Y.max()
    
  
    nb_features = int(n_feature/timestep)
    

    train_num_all = sum(train_num_array)    
    
    X_train = np.zeros((train_num_all,timestep,nb_features))
    
    X_test = np.zeros((sum(Y>0)[0]-train_num_all,timestep,nb_features));
  
    
    X_PCA = featureNormalize(PCANorm(X,num_PC),2)    

    X = featureNormalize(X,1)

    
    hw = int(w/2)
    
    X_PCAMirrow = MirrowCut(X_PCA.reshape(row,col,num_PC),hw)
 
    XP = np.zeros((K,w,w,num_PC))
    
    for i in range(1,K+1):
        index_row = int(np.ceil(i*1.0/col))
        index_col = i - (index_row-1)*col + hw -1 
        index_row += hw -1
        patch = X_PCAMirrow[index_row-hw:index_row+hw,index_col-hw:index_col+hw,:]
        XP[i-1,:,:,:] = patch
    
    #XP = np.moveaxis(XP, 3, 1)


    if israndom==True:
        randomArray = list()
        for i in range(1,n_class+1):
            index = np.where(Y==i)[0]
            n_data = index.shape[0]
            randomArray.append(np.random.permutation(n_data))
  

    flag1=0
    flag2=0
    
    X_train = np.zeros((train_num_all,timestep,nb_features))
    XP_train = np.zeros((train_num_all,w,w,num_PC))  
    #XP_train = np.zeros((train_num_all,num_PC,w,w))  
    Y_train = np.zeros((train_num_all,1))
    
    X_test = np.zeros((sum(Y>0)[0]-train_num_all,timestep,nb_features))
    XP_test = np.zeros((sum(Y>0)[0]-train_num_all,w,w,num_PC))    
    #XP_test = np.zeros((sum(Y>0)[0]-train_num_all,num_PC,w,w))   
    Y_test = np.zeros((sum(Y>0)[0]-train_num_all,1))   
    
    
    
    for i in range(1,n_class+1):
        index = np.where(Y==i)[0]
        n_data = index.shape[0]
        train_num = train_num_array[i-1]
        randomX = randomArray[i-1]
        

        XP_train[flag1:flag1+train_num,:,:,:] = XP[index[randomX[0:train_num]],:,:,:]
        Y_train[flag1:flag1+train_num,0] = Y[index[randomX[0:train_num]],0]
            
        XP_test[flag2:flag2+n_data-train_num,:,:,:] = XP[index[randomX[train_num:n_data]],:,:,:]
        Y_test[flag2:flag2+n_data-train_num,0] = Y[index[randomX[train_num:n_data]],0]
            
        if s1s2==2:
            
            for j in range(0,timestep):
                X_train[flag1:flag1+train_num,j,:] = X[index[randomX[0:train_num]],j:j+(nb_features-1)*timestep+1:timestep]
                X_test[flag2:flag2+n_data-train_num,j,:] = X[index[randomX[train_num:n_data]],j:j+(nb_features-1)*timestep+1:timestep]
                
        else:
            for j in range(0,timestep):
                X_train[flag1:flag1+train_num,j,:] = X[index[randomX[0:train_num]],j*nb_features:(j+1)*nb_features]
                X_test[flag2:flag2+n_data-train_num,j,:] = X[index[randomX[train_num:n_data]],j*nb_features:(j+1)*nb_features]
                
        flag1 = flag1+train_num
        flag2 = flag2+n_data-train_num
        
        
    X_reshape = np.zeros((X.shape[0],timestep,nb_features))
    if s1s2==2:
        
        for j in range(0,timestep):
            X_reshape[:,j,:] = X[:,j:j+(nb_features-1)*timestep+1:timestep]
    else:
        for j in range(0,timestep):
            X_reshape[:,j,:] = X[:,j*nb_features:(j+1)*nb_features]
            

    X = X_reshape    
    
    return X.astype('float32'),X_train.astype('float32'),X_test.astype('float32'),XP.astype('float32'),XP_train.astype('float32'),XP_test.astype('float32'),Y.astype(int),Y_train.astype(int),Y_test.astype(int)

   
   
   