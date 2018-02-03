
# coding: utf-8

# In[ ]:


from keras.layers import Input, Dense
from keras.layers.core import Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, RMSprop
import os
import os.path
import numpy as np
from PIL import Image
from numpy import * 
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import scipy.misc
import cv2


# In[ ]:


input_img = Input(shape=(128,128,1))
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
#enco = MaxPooling2D(pool_size=(2, 2))(enco)
    
#enco = Conv2D(256, (3, 3), activation='relu', padding='same')(enco)
#enco = BatchNormalization()(enco)
#enco = Conv2D(256, (3, 3), activation='relu', padding='same')(enco)
#enco = BatchNormalization()(enco)


# In[ ]:


#deco = UpSampling2D((2,2))(enco)
#deco = Conv2D(128, (3, 3), activation='relu', padding='same')(deco)
#deco = BatchNormalization()(deco)
#deco = Conv2D(128, (3, 3), activation='relu', padding='same')(deco)
#deco = BatchNormalization()(deco)

deco = UpSampling2D((2,2))(enco)
deco = Conv2D(64, (3, 3), activation='relu', padding='same')(deco)
deco = BatchNormalization()(deco)
deco = Conv2D(64, (3, 3), activation='relu', padding='same')(deco)
deco = BatchNormalization()(deco)

deco = UpSampling2D((2,2))(deco)
deco = Conv2D(32, (3, 3), activation='relu', padding='same')(deco)
deco = BatchNormalization()(deco)
deco = Conv2D(32, (3, 3), activation='relu', padding='same')(deco)
deco = BatchNormalization()(deco)
   
deco = UpSampling2D((2,2))(deco)
deco = Conv2D(16, (3, 3), activation='relu', padding='same')(deco)
deco = BatchNormalization()(deco)
deco = Conv2D(16, (3, 3), activation='relu', padding='same')(deco)
deco = BatchNormalization()(deco)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(deco)
    


# In[ ]:


autoencoder = Model(input_img, decoded)


# In[ ]:


encoder = Model(input_img, enco)


# In[ ]:


ada=Adadelta(lr=5.0, rho=0.95, epsilon=1e-08, decay=0.001)
rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
autoencoder.compile(loss='mean_squared_error', optimizer=rms)
encoder.compile(loss='mean_squared_error', optimizer=rms)

basic_mat=[]
tobe_mat=[]


# In[ ]:


path1="Data"


# In[ ]:


for i in range(1,124):
    path_major=path1+'/'+str(i)
    for j in range(1,101):
        img=array(Image.open(path_major+"/"+str(j)+"_.jpg"))
        #print shape(img)
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        img=img.reshape(128,128,1)
        basic_mat.append(img)
        tobe_mat.append(img)


# In[ ]:


data,Label = shuffle(basic_mat,tobe_mat, random_state=2)


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

y_train = y_train.astype('float32') / 255.
y_test = y_test.astype('float32') / 255.
CheckDir = 'sample/'


# In[ ]:


for epoch in range(1,300):
    
    train_X,train_Y=shuffle(x_train,y_train)
    print ("Epoch is: %d\n" % epoch)
    batch_size=64
    print ("Number of batches: %d\n" % int(len(train_X)/batch_size))
    num_batches=int(len(train_X)/batch_size)
    for batch in range(num_batches):    
        batch_train_X=train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
        batch_train_Y=train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]
        loss=autoencoder.train_on_batch(batch_train_X,batch_train_Y)
        print ('epoch_num: %d batch_num: %d loss: %f\n' % (epoch,batch,loss))

    autoencoder.save_weights("fabric_autoen_final_500.h5")
    encoder.save_weights("Only_Encoder_500.h5")
    if(epoch%1==0):
        x_test,y_test=shuffle(x_test,y_test)
        decoded_imgs=autoencoder.predict(x_test[:2])
        temp = np.zeros([128, 128*3,3])
        temp[:, :128,:1] = x_test[0,:,:,:1]
        temp[:, 128:128*2,:1] = y_test[0,:,:,:1]
        temp[:, 128*2:,:1] = decoded_imgs[0,:,:,:1]
        temp[:,:,1]=temp[:,:,0]
        temp[:,:,2]=temp[:,:,0]
        temp = temp*255
        scipy.misc.imsave(CheckDir + str(epoch) + ".jpg", temp)

