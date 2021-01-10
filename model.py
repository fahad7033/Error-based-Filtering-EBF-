from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import time

from NoiseAdding import *
from helperFunctions import *





# setting of initial parameters
batch_size = 512  # batch size during EBF implementation
num_classes = 10
addNoise = True
EBF = True
num_rounds = 3           #number of rounds of EBF
noise_type = 'symmetric' # noise type 'pairflip' or 'symmetric'
noise_rate = 0.2         #noise ratio





# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

    
    
    


if(addNoise):
    x_train, y_temp = corruptingData(x_train, y_train, noise_type , noise_rate , num_classes )
    
    y1 = keras.utils.to_categorical(y_temp[:,0], num_classes)
    y2 = y_temp[:,1].reshape(y_temp.shape[0],1)
    y_train = np.append(y1, y2, axis=1)
    print('y shape: ',y_train.shape)
else:    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.01)

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation=lrelu, input_shape=input_shape, strides=(1,1), padding='same'))
model.add(Conv2D(128, (3, 3), activation=lrelu, strides=(1,1), padding='same' ))
model.add(Conv2D(128, (3, 3), activation=lrelu, strides=(1,1), padding='same' ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation=lrelu, strides=(1,1), padding='same' ))
model.add(Conv2D(256, (3, 3), activation=lrelu, strides=(1,1), padding='same' ))
model.add(Conv2D(256, (3, 3), activation=lrelu, strides=(1,1), padding='same' ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), activation=lrelu, strides=(1,1), padding='same'))
model.add(Conv2D(256, (3, 3), activation=lrelu, strides=(1,1), padding='same'))
model.add(Conv2D(128, (3, 3), activation=lrelu, strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(AveragePooling2D())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))


opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer= opt,
              metrics=['accuracy'])




## printing the second method with train_on_batch
# calculate the number of batches per training epoch
bat_per_epo = int(x_train.shape[0] / batch_size)
# calculate the number of training iterations
n_steps = bat_per_epo * epochs
# calculate the size of half a batch of samples
half_batch = int(batch_size / 2)


NoiseDic = []
totalerrors = []
eliminatedInstCount = list()
eliminatedInstances = list()
c = 0
filteringCount = 0






trainX_batches, trainY_batches = generate_batches(x_train, y_train, batch_size)
# manually enumerate epochs
print("Training has been started...")
for epoch in range(50):
    loss = 0  
    totalLoss = 0
    sampelsCount = 0
    
#    for i in range(bat_per_epo):
    for batch in range(len(trainX_batches)):
        if (batch< len(trainX_batches)):
            trainX_batch = trainX_batches[batch]
            trainY_batch = trainY_batches[batch]
        else:
            break;

      

        
        l = model.train_on_batch(trainX_batch, trainY_batch[:,0:10])
        loss += l[0]
                
        sampelsCount += len(trainX_batch)
        print('\r' + '{}'.format(sampelsCount) + '/{}'.format(x_train.shape[0]), end='')
        time.sleep(1)
    

        if(EBF):
                    
            if (epoch>0):
                c +=1
            
            if (epoch>0 and filteringCount<num_rounds and c%16==0):
                print(">>>>>> Batch No. : ", c)
                y_pred = model.predict(x_train)
                y_true = y_train[:,0:10].astype('float32')
                y_true = K.constant(y_true)
                y_pred = K.constant(y_pred)
                g = K.categorical_crossentropy(target=y_true, output=y_pred)
                ce = K.eval(g)  # 'ce' for cross-entropy
                type(ce)
                    
                print("$1: ",np.shape(totalerrors))
                totalerrors = collectErrors(totalerrors, ce)
                print("$2: ",np.shape(totalerrors))
        
                if c == 160: 
                    x_train, y_train, NoiseDic, elimiInst = filteringEx(NoiseDic, x_train, y_train, totalerrors, epoch)                
                    trainX_batches, trainY_batches = generate_batches(x_train, y_train, batch_size)  
                    totalerrors = []              
                    eliminatedInstCount.append(len(elimiInst))
                    eliminatedInstances.extend(elimiInst)
                    print('... Eliminated: ',len(elimiInst))
                    print('... Still     : ', x_train.shape[0])
                    c = 0
   
                    filteringCount +=1
 
    score = model.evaluate(x_test, y_test, verbose=0)
    score0 = model.evaluate(x_train, y_train[:,0:10], verbose=0)
    print('\n>%d , loss:%.3f, accuracy:%.4f || Test loss: %.3f , Test Accuracy: %.4f'% ((epoch+1), score0[0], score0[1], score[0] , score[1]))
    
    if filteringCount==num_rounds:
        print("Filtering procedure has been completed...")
        break
    
    
    
    #############################################################################################################
    ######################        Augmentation process
    #####################################################################################################
print("\n\n\n Training with Augmentation procedure has been started...")
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=True,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=25,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    
datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
y = y_train[:,0:10]
batch_size=128
epochs = 200 - epoch -1
model.fit_generator(datagen.flow(x_train, y,
                                 batch_size=batch_size), steps_per_epoch=int(len(x_train)/batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    workers=4)    

    
        
   
        
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


