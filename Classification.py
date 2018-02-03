
# coding: utf-8

# In[ ]:


from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.core import Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, RMSprop
from keras import backend as K
from keras.models import model_from_json
import os
import os.path
import numpy as np
from PIL import Image
from numpy import * 
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import scipy.misc
import cv2
import theano
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


input_img = Input(shape=(128,128,1))
num_classes = 123
img_rows, img_cols = 128, 128


# In[ ]:


enco = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
enco = BatchNormalization()(enco)
enco = Conv2D(16, (3, 3), activation='relu', padding='same')(enco)
enco = BatchNormalization()(conv1)
enco = MaxPooling2D(pool_size=(2, 2))(enco)
   
enco = Conv2D(32, (3, 3), activation='relu', padding='same')(enco)
enco = BatchNormalization()(enco)
enco = Conv2D(32, (3, 3), activation='relu', padding='same')(enco)
enco = BatchNormalization()(enco)
enco = MaxPooling2D(pool_size=(2, 2))(enco)
      
enco = Conv2D(64, (3, 3), activation='relu', padding='same')(enco)
enco = BatchNormalization()(enco)
enco = Conv2D(64, (3, 3), activation='relu', padding='same')(enco)
enco = BatchNormalization()(enco)
enco = MaxPooling2D(pool_size=(2, 2))(enco)
    
enco = Conv2D(128, (3, 3), activation='relu', padding='same')(enco)
enco = BatchNormalization()(enco)
enco = Conv2D(128, (3, 3), activation='relu', padding='same')(enco)
enco = BatchNormalization()(enco)
enco = MaxPooling2D(pool_size=(2, 2))(enco)
    
enco = Conv2D(256, (3, 3), activation='relu', padding='same')(enco)
enco = BatchNormalization()(enco)
enco = Conv2D(256, (3, 3), activation='relu', padding='same')(enco)
enco = BatchNormalization()(enco)

enco.load_weights("Only_Encoder.h5")

classify = Flatten()(enco)
classify = Dense(64, activation='relu')(classify)
classify = Dense(32, activation='relu')(classify)
classify = Dense(num_classes, activation='softmax')(classify)

network = Model(enco, classify)
final_network = Model(input_img, classify)
rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
network.compile(loss='mean_squared_error', optimizer=rms)
final_network.compile(loss='mean_squared_error', optimizer=rms)


# In[ ]:


path="Data"
basic_mat=[]
index=[]
epochs=100
batch_size1=64


# In[ ]:


for i in range(1,124):
    path_major=path1+'/'+str(i)
    for j in range(1,101):
        img=array(Image.open(path_major+"/"+str(j)+"_.jpg"))
        #print shape(img)
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        img=img.reshape(128,128,1)
        basic_mat.append(img)
        #tobe_mat.append(img)
    #index[i]=i
    index.append(i)


# In[ ]:


data,Label = shuffle(basic_mat,index, random_state=2)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, Label, test_size=0.2, random_state=2)
X_train = array(X_train)
y_train = array(y_train)
X_test = array(X_test)
y_test = array(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


x_train = X_train.astype('float32') / 255.
x_test = X_test.astype('float32') / 255.
network.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=batch_size1, nb_epoch=epochs, verbose=2)
scores = network.evaluate(x_test, y_test, verbose=0)
print ("%s: %.2f%%" % (network.metrics_names[1], scores[1]*100))
network.save_weights("only_classify.h5")
final_network.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=batch_size1, nb_epoch=epochs, verbose=2)
scores_final = final_network.evaluate(x_test, y_test, verbose=0)
print ("%s: %.2f%%" % (final_network.metrics_names[1], scores_final[1]*100))
final_network.save_weights("complete_model.h5")

