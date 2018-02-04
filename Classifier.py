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
import matplotlib.pyplot as plt
import matplotlib


input_img = Input(shape=(128,128,1))
num_classes = 123
img_rows, img_cols = 128, 128


enco = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
enco = BatchNormalization()(enco)
enco = Conv2D(16, (3, 3), activation='relu', padding='same')(enco)
enco = BatchNormalization()(enco)
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

#enco.load_weights("Only_Encoder.h5")
encoder = Model(input_img, enco)    
encoder.load_weights("Only_Encoder.h5")

classify = Flatten()(enco)
classify = Dense(64, activation='relu')(classify)
classify = Dense(32, activation='relu')(classify)
classify = Dense(num_classes, activation='softmax')(classify)

#network = Model(enco, classify)
network = Model(input_img, classify)
rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
#network.compile(loss='mean_squared_error', optimizer=rms)
network.compile(loss='mean_squared_error', optimizer=rms)
#network.summary()
for layers in encoder.layers:
	layers.trainable=False
	
	
basic_mat=[]
#tobe_mat=[]
#num_classes = 123
lab=[]
