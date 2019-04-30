#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 23:05:58 2019

@author: aviallon
"""
import os
import time

#import plaidml.keras
#plaidml.keras.install_backend()

import keras
import numpy as np
import matplotlib.pyplot as plt

from keras import applications
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
import keras.preprocessing.image as preprocess

#np.random.seed()
#model = applications.VGG16(include_top=False, weights='imagenet')

batch_size = 24
num_classes = 10
epochs = 100
data_augmentation = True
resume = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_denoiser_model-'+str(int(time.time()))[-6:] # Add a timestamp at the end to avoid overwriting


# The data, split between train and test sets:
(y_train, temp), (y_test, temp2) = cifar100.load_data()


def add_noise(x):
    return np.clip(np.random.poisson(x), 0, 255)

def compare_im(image, lignes=1, n=0):
    datagen = preprocess.ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        preprocessing_function=None,
        validation_split=0.0)
    
    
    datagen.fit([image])
    noisy = add_noise(image*255) / 255
    forme = image.shape
    reshape = (1, forme[0], forme[1], forme[2])
    predict = np.clip(model.predict_generator(generator=datagen.flow(np.reshape(noisy, reshape), np.reshape(image, reshape)))[0], 0, 1)
    orig = image
    plt.subplot(lignes,3,(3*n+1))
    plt.imshow(noisy)
    plt.axis('off')
    plt.subplot(lignes,3,(3*n+2))
    plt.imshow(predict)
    plt.axis('off')
    plt.subplot(lignes,3,(3*n+3))
    plt.imshow(orig)
    plt.axis('off')
    
def sample_images(images):
    for i, image in enumerate(images):
        compare_im(image, len(images), i)
    plt.show()
    
def load_image(path):
    return plt.imread(path, format='rgb')/255

def predict(image):
    forme = image.shape
    reshape = (1, forme[0], forme[1], forme[2])
    return np.clip(model.predict_generator(generator=datagen.flow(np.reshape(image, reshape), np.reshape(image, reshape)))[0], 0, 1)

x_train, x_test = add_noise(y_train), add_noise(y_test)

y_train = y_train.astype('float32')
y_train /= 255
y_test = y_test.astype('float32')
y_test /= 255
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

input_dim = x_test.shape[1:]
output_dim = y_test.shape[1:]

# Yep, that data generator actually does nothing... it is just used to flow training data
datagen = preprocess.ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False,
    preprocessing_function=None,
    validation_split=0.0)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train) # That is the useful part of the generator actually

model = Sequential()

model.add(Conv2D(20, (5, 5), padding='same', input_shape=(None, None, 3)))
model.add(LeakyReLU(alpha=0.01))
model.add(Conv2DTranspose(3, (5, 5), padding='same'))
model.add(Activation('relu'))

opt = keras.optimizers.Nadam()
model.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mean_squared_error'])

print(model.summary())

stop_when_no_improvements = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

checkpoint = ModelCheckpoint("model.h5", monitor="val_loss", verbose=1, save_best_only=True)

if resume:
    try:
        if os.path.isfile('model.h5'):
            model.load_weights('model.h5')
    except ValueError:
        os.rename('model.h5', 'model.h5.old')
else:
    print('Not resuming previous learn.')

# fits the model on batches with real-time data augmentation:
history = model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                    epochs=epochs,
                    workers=16,
                    callbacks = [checkpoint, stop_when_no_improvements],
                    validation_data = (x_test, y_test))


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name+'.h5')
model.save(model_path)
with open(os.path.join(save_dir, model_name+'.json'), 'w') as f:
    f.write(model.to_json())
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
compare_im(y_test[0])