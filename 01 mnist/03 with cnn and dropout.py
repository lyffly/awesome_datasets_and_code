#### coding by 刘云飞


from keras.datasets import mnist
from keras import models,layers
from keras.utils import to_categorical,plot_model
import time

(train_imgs,train_labels),(test_imgs,test_labels) = mnist.load_data()

t_imgs=train_imgs.reshape((60000,28,28,1))
t_imgs=t_imgs.astype('float32')/255
test_imgs=test_imgs.reshape((10000,28,28,1))
test_imgs=test_imgs.astype('float32')/255

t_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)


model = models.Sequential()
model.add(layers.Conv2D(64, 2,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(layers.Conv2D(128,2,padding='same',activation='relu'))
model.add(layers.Conv2D(128,2,padding='same',activation='relu'))
model.add(layers.Conv2D(128,2,padding='same',activation='relu'))
model.add(layers.Conv2D(64,2,padding='same',activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10,activation='softmax'))


model.summary()
plot_model(model,to_file='with cnn and dropout.png')

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(t_imgs,t_labels,epochs=5,batch_size=128)


test_loss, test_acc = model.evaluate(test_imgs, test_labels)
print(test_acc)

