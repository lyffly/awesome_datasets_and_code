#coding=utf-8
# coding by 刘云飞
# 2018-01-12
import os
import numpy as np
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


model =models.Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.load_weights("no1.h5")

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=0.0001),metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,
                width_shift_range=0.2,height_shift_range=0.2,
                shear_range=0.2,zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "D:/Project/B1/data_small/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    "D:/Project/B1/data_small/validation",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

history =  model.fit_generator(train_generator,
            steps_per_epoch =100,
            epochs = 60,
            validation_data=validation_generator,
            validation_steps=20)

model.save("no1.h5")

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss =  history.history['loss']
val_loss =  history.history['val_loss']

epochs = range(1,len(acc) +1)

plt.plot(epochs,acc,'bo',label='Training_acc')
plt.plot(epochs,val_acc,'b',label='validation_acc')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training_loss')
plt.plot(epochs,val_loss,'b',label='validation_loss')
plt.legend()

plt.show()

