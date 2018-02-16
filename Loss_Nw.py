import os
import cv2
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from normalized_correlation_layer import Normalized_Correlation_Layer

def Feature_Extractor(input_shape):
	convnet = Sequential()
	convnet.add(Conv2D(16,(9,9),activation='relu',padding='same',input_shape=input_shape))
	convnet.add(MaxPooling2D())
	convnet.add(Conv2D(32,(7,7),activation='relu',padding='same'))
	convnet.add(MaxPooling2D())
	convnet.add(Conv2D(64,(5,5),activation='relu',padding='same'))
	convnet.add(MaxPooling2D())
	convnet.add(Conv2D(128,(3,3),activation='relu',padding='same'))
	convnet.add(MaxPooling2D())
	convnet.add(Conv2D(256,(3,3),activation='relu',padding='same'))
	convnet.add(Conv2D(512,(3,3),activation='sigmoid',padding='same'))
	return convnet
	
'''def Loss_NW(input_shape):
	left_input = Input(input_shape)
	right_input = Input(input_shape)
	input_combined = [left_input,right_input]
	y_correlated = Normalized_Correlation_Layer()([left_input,right_input])
	outp = Conv2D(128,(3,3),activation='relu',padding='same')(y_correlated)
	outp=MaxPooling2D()(outp)
	outp = Conv2D(256,(3,3),activation='relu',padding='same')(outp)
	outp=Flatten()(outp)
	outp=Dropout(0.3)(outp)
	outp=Dense(2000, activation = 'relu')(outp)
	outp=Dropout(0.3)(outp)
	outp=Dense(1000, activation = 'relu')(outp)
	outp=Dropout(0.3)(outp)
	outp=Dense(100, activation = 'relu')(outp)
	outp=Dropout(0.3)(outp)
	outp=Dense(1, activation = 'sigmoid')(outp)
	Network=Model(input=input_combined, output = outp)
	return Network'''
	
def Loss_NW(input_shape):
	left_input = Input(input_shape)
	right_input = Input(input_shape)
	input_combined = [left_input,right_input]
	convnet = Sequential()
	convnet.add(Flatten(input_shape = input_shape))
	#convnet.add(Dropout(0.3))
	#convnet.add(Dense(2048,activation="sigmoid"))
	encoded_l = convnet(left_input)
	encoded_r = convnet(right_input)
	#merge two encoded inputs with the l1 distance between them
	L1_distance = lambda x: K.abs(x[0]-x[1])
	#May think of another distance
	both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
	both = Dropout(0.3)(both)
	prediction=Dense(100,activation='relu')(both)
	prediction = Dropout(0.3)(prediction)
	prediction = Dense(1, activation = 'sigmoid')(prediction)


	Network=Model(input=input_combined, output = prediction)
	return Network	

input_shape = (128, 128, 3)
Cnn_network = Feature_Extractor(input_shape)
Cnn_network.load_weights("../Models/CNN_3_channel.h5")
middle_shape = (8,  8, 512)
L_network = Loss_NW(middle_shape)

left_input = Input((128,128,3))
right_input = Input((128,128,3))
input_combined = [left_input, right_input]

left_encoding = Cnn_network(left_input)
right_encoding = Cnn_network(right_input)

prediction = L_network([left_encoding, right_encoding])

Final_Model = Model(input = input_combined, output = prediction)

'''for layers in Cnn_network.layers:
	layers.trainable = False'''

Final_Model.summary()
#exit()
lr = 0.00006
optimizer = Adam(lr)
Final_Model.compile(loss="mean_squared_error",optimizer=optimizer)
L_network.load_weights("../Models/Loss_NW.h5")
images_home_path = "../Db_c/2/"
images_home_path = "../Db_c/2/"
print 'Loading Data...............'
images=[]
for s in range(1,387):
	for p in range(1,21):
		image_path = images_home_path + 's'+str(s)+'/'+str(p)+'_IRT_N.bmp'
		img = cv2.imread(image_path)
		img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
		img = img.reshape(128,128,1)
		img = img.astype('float32') / 255.
		image_path2 = images_home_path + 's'+str(s)+'/'+str(p)+'.bmp'
		img2 = cv2.imread(image_path2)
		img2 = cv2.cvtColor( img2, cv2.COLOR_RGB2GRAY )
		img2 = img2.reshape(128,128,1)
		img2 = img2.astype('float32') / 255.
		image_path3 = images_home_path + 's'+str(s)+'/0'+str(p)+'_TCM_N.bmp'
		img3 = cv2.imread(image_path3)
		img3 = cv2.cvtColor( img3, cv2.COLOR_RGB2GRAY )
		img3 = img3.reshape(128,128,1)
		img3 = img3.astype('float32') / 255.
		image = np.zeros([128,128,3])
		image[:,:,2:3] = img
		image[:,:,0:1] = img2
		image[:,:,1:2] = img3
		images.append(image)
print '----------------------Data Loaded------------------------------'

def make_batch2(s1,p1,n,R):                                        #random logic specifically for batch size of > 8 and of pow 2
    x = 128
    y = 128
    pairs = [np.zeros((n, x, y,3)) for i in range(2)]
    targets = np.zeros((n,))
    i = R
    for j in range(R):
        pairs[0][j,:,:,:]=images[(s1-1)*20+p1-1].resh                                                                               ape(x,y,3)
        pairs[1][j,:,:,:]=images[(s1-1)*20+(p1-1+j)%11].reshape(x,y,3)
        targets[j] = 0
    while (i < n):
        #if i<2:                                                #  total R positive samples
        if i < min(2,R):                                        #  at max first 2 normal positives
            for p2 in range (1,11):                              
                if p2 != p1:                                    
                    pairs[1][i,:,:,:] = images[(s1-1)*20+p2-1]
                    targets[i] = 0
                    pairs[0][i,:,:,:] = images[(s1-1)*20+p1-1]
                    i += 1
                    if R==1: break                              # if R==1, add only one normal positive
        # elif i<n/4 :                                          #  rest R-2 random positive
        elif i < R:
            s2 = s1
            while (s2 == s1) :
                s2=rng.randint(1,387)
            p2 = p1
            while p2 == p1 :
                p2=rng.randint(1,11)
            pairs[0][i,:,:,:] = images[(s2-1)*20+p1-1]
            pairs[1][i,:,:,:] = images[(s2-1)*20+p2-1]
            targets[i]=0 
            i += 1
        else :                                                  # N-R random negative samples
            s2 = s1
            while (s2 == s1) :
                s2=rng.randint(1,387)
            pairs[0][i,:,:,:] = images[(s1-1)*20+p1-1].reshape(x,y,3)
            pairs[1][i,:,:,:] = images[(s2-1)*20+p1-1].reshape(x,y,3)
            targets[i] = 1
            i += 1
    return pairs,targets
def make_batch(s1,p1,n):
	pairs=[np.zeros((n, 8, 8, 512)) for i in range(2)]
	targets=np.zeros((n,))
	for i in range(0,10): targets[i] = 0
	for i in range(10,n): targets[i] = 1
	for i in range(n):
		pairs[0][i,:,:,:] = Cnn_network.predict(images[(s1-1)*20+p1-1].reshape(1, 128, 128, 1))
	#for i in range(3):

	for i in range(10):
		p=p1
		while(p==p1):
			p=rng.randint(1,10)
		pairs[1][i,:,:,:] = Cnn_network.predict(images[(s1-1)*20+i].reshape(1, 128, 128, 1))
	for i in range(10,n):
		s2=s1 
		while(s2==s1):
			s2=rng.randint(1,386)
		p2=rng.randint(1,10)
		pairs[1][i,:,:,:] = Cnn_network.predict(images[(s2-1)*20+p2-1].reshape(1, 128, 128, 1))
	return pairs,targets
	
Iterations = 200
N = 32
output_file_name = "Loss_NW.txt"

counter_array = [((x+1)%386) for x in range(387)]

b = []
while(len(b) < 16000):
    a = rng.gamma(4, 2.67,20000)
    c = filter(lambda x: (x>=0 and x<=32),map(lambda x: int(x), np.rint(a)))
    b = [int((x*N)/32) for x in c]
    
currentPointer = -1

ratio_update_counter,numPositives = [0 for x in range(3)],[4 for x in range(3)]

def getNumPositives(currentPointer):
    currentPointer += 1 
    return b[currentPointer] , currentPointer
	
for i in range(1,Iterations+1):
	print('Iteration '+str(i)+' started')
	# loss = [0 for x in range(3)]
	train_loss = 0.0
	for s in range(1,387):											###387
		for p in range(1,11):										##11
			ratio_update_counter[0] = (ratio_update_counter[0] + 1) % 60
			if ratio_update_counter[0]==0: numPositives[0],currentPointer = getNumPositives(currentPointer)
			pairs,label = make_batch(s,p,N,numPositives[0])
			current_loss = Final_Model.train_on_batch(pairs,label)
			train_loss += current_loss
	L_network.save_weights("../Models/Loss_NW_2.h5")
	Cnn_network.save_weights("../Models/CNN_3_channel_2.h5")
	print 'Weights Saved'
    # take first 10 subjects 5th pose, match with 1-4 of first 25, save loss
	test_loss = 0.0
	for ijk in xrange(10):
		probe_image = Cnn_network.predict(images[ijk*20+18].reshape(1, 128, 128, 3))
		for j in xrange(10):
			for k in xrange(15):
				gallery_image = Cnn_network.predict(images[j*20+k].reshape(1, 128, 128, 3))
				score = L_network.predict([probe_image,gallery_image])
				if j==ijk: test_loss += score
				else: test_loss += (1.0-score)

    # test end 
	print 'iteration ' + str(i) + ' training loss: ' + str(train_loss) + ' test_loss: ' + str(test_loss[0][0])
	with open(output_file_name,'a+') as f:
		f.write('Iteration ' + str(i) + '\ttraining loss ' + str(train_loss) + '\ttest_loss ' + str(test_loss[0][0]) + '\n')
