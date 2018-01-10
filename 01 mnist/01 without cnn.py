#1. load mnist dataset in Keras
from keras.datasets import mnist
(train_imgs,train_labels),(test_imgs,test_labels) = mnist.load_data()

#2. the training data 
train_imgs.shape
len(train_imgs)
#3. the test data
#4. the network
from keras import models,layers
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

#5. the compilation step
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#6. preparing the image data
t_imgs=train_imgs.reshape((60000,28*28))
t_imgs=t_imgs.astype('float32')/255
test_imgs=test_imgs.reshape((10000,28*28))
test_imgs=test_imgs.astype('float32')/255

#7. preparing the labels
from keras.utils import to_categorical
t_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#8. training the network
network.fit(t_imgs,t_labels,epochs=5,batch_size=128)

#9. evaluating the network
test_loss, test_acc = network.evaluate(test_imgs, test_labels)

print(test_acc)