#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 23:05:58 2019

@author: aviallon
"""
import os
import time
import argparse

#import plaidml.keras
#plaidml.keras.install_backend()

batch_size = 40
epochs = 100
#data_augmentation = True
resume = 0
try:
    print(data_already_peprocessed)
except NameError:
    data_already_peprocessed = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_denoiser_model-'+str(int(time.time()))[-6:] # Add a timestamp at the end to avoid overwriting
noises = ['poisson', 'gaussian', 'salt', 'wavelet', 'inpainting', 'highiso']

parser = argparse.ArgumentParser(description='Train denoising models.')
parser.add_argument('--noise', dest='noises', choices=noises, nargs='+', default=['poisson', 'gaussian'], required=False, help='specify on which noises we should train')
parser.add_argument('--name', help='output network name')
parser.add_argument('--dataset', default='cifar100', choices=['cifar100', 'data'], help='dataset on which to train')
parser.add_argument('--resume', default=0, type=int, help='resume last training if it exists')
parser.add_argument('--batch_size', dest='bsize', default=40, type=int, help='set batch size')
parser.add_argument('--architecture', dest='arch', default='simple', help='choose network architecture')
parser.add_argument('--history', default=0, type=int, help='display training history at the end')
parser.add_argument('--opencl', default=1, type=int, help='use PlaidML as backend')

args = parser.parse_args()
print(args)
if args.name != None:
    model_name = args.name
if args.noises != None:
    noises = args.noises
resume = args.resume
batch_size = args.bsize

checkpoint_name = "model.h5"

dataset = args.dataset

if args.opencl:
    import plaidml.keras
    plaidml.keras.install_backend()
    checkpoint_name = "model-ocl.h5"

import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt

import keras_contrib
from keras import applications
from keras.datasets import cifar100
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
import keras.preprocessing.image as preprocess

from multiprocessing import Pool

np.random.seed()
#model = applications.VGG16(include_top=False, weights='imagenet')

def load_image(path, divide=255.0):
    #print(img.shape)
    return (plt.imread(path, format='rgb')[:,:,:3]/divide)

def crop(x, img_size, rand=1):
    #img_size = 512
    if x.shape[0] > img_size and x.shape[1] > img_size:
        xrand, yrand = 0, 0
        if rand != 0:
            xrand = np.random.randint(-rand, rand)
            yrand = np.random.randint(-rand, rand)
        xm = int(x.shape[0]//2-(img_size/2)+xrand)
        ym = int(x.shape[1]//2-(img_size/2)+yrand)
        return x[xm:xm+img_size, ym:ym+img_size, :]
    else:
        raise ResourceWarning('Image too small ({}, {}), passing'.format(x.shape[0], x.shape[1]))

data_dir = 'data'
    
if dataset == 'cifar100':
    (y_train, temp), (y_test, temp2) = cifar100.load_data()

#print("Loading images into memory...", end='', flush=True)
#y_train = []
#y_test = []
#for f in os.scandir(data_dir):
#    y_train.append(load_image(f.path, divide=1))
#    
#deb = -int(len(y_train)*0.1)-1
#y_test = y_train[deb:]
#del(y_train[deb:])
#y_train = np.array(y_train)
#y_test = np.array(y_test)
#print(" done !", flush=True)

def add_poisson_noise(x):
    return np.random.poisson(x)

def add_gaussian_noise(x):
    return x + np.random.normal(scale=15, size=x.shape)

def _add_wavelet_noise(x):    
    gris = np.random.randint(-30, 30, dtype=int, size=(x.shape[0], x.shape[1]))
    
    return x + np.stack((gris, gris, gris), axis=2)

def add_wavelet_noise(x, multiprocessing):
    if len(x.shape) > 3:
        if multiprocessing:
            with Pool(16) as p:
                x_new = p.map(_add_wavelet_noise, x)
        else:
            x_new = np.array([_add_wavelet_noise(im) for im in x])
        
        return x_new
    else:
        return _add_wavelet_noise(x)

def _add_salt_noise(x):
    #x = image.copy()

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if np.random.rand() > 0.8:
                x[i,j] = [255]*3
    return x

def add_salt_noise(x, multiprocessing):
    if len(x.shape) > 3:
        if multiprocessing:
            with Pool(16) as p:
                x_new = p.map(_add_salt_noise, x)
        else:
            x_new = np.array([_add_salt_noise(im) for im in x])
        
        return x_new
    else:
        return _add_salt_noise(x)
    
def _add_blue_hole(img):
    from numpy.random import randint
    #color = (0, 244, 238)
    color = (0, 0, 0)
    return np.array(cv2.circle(img, (randint(0,img.shape[0]), randint(0,img.shape[1])), randint(img.shape[1]//35, img.shape[1]//15), color, -1))
    
def add_blue_hole(x):
    #xnew = x.copy()
    
    if len(x.shape) > 3:
        return np.array([_add_blue_hole(im) for im in x])
    else:
        return _add_blue_hole(x)
    
noise_image = (plt.imread('../datasets/noise/101HPIMG/HPIM3024.JPG')[:,:,:3])*9
noise = cv2.repeat(crop(noise_image, 300, 64), 15, 15)
def _add_high_iso_noise(x):
    if x.shape[0] > noise.shape[0] or x.shape[1] > noise.shape[1]:
        raise Exception('Image too big for HIGH ISO noise')
    
    return np.array(x + crop(noise, x.shape[0], min(1, abs(noise.shape[0]//2-x.shape[0]))))

def add_high_iso_noise(x):
    if len(x.shape) > 3:
        return np.array([_add_high_iso_noise(im) for im in x])
    else:
        return _add_high_iso_noise(x)
    
def add_noise(img, multiprocessing=False, adapt = True, batch = False):
    x = img.copy()
    #print(x.shape, x.dtype)
    if len(x.shape) > 3 or batch:
        x = x[:, :, :, :3]
    elif len(x.shape) == 2:
        x = np.stack((x, x, x), axis=2)
    else:
        x = x[:, :, :3]
    if (x.dtype == float or x.dtype == 'float32' or x.dtype == 'float64' or np.max(x) <= 1) and adapt:
        #x *= 255
        x = (x * 255).astype(int)
        print(x)
        print('dividing')
    if 'poisson' in noises:
        x = add_poisson_noise(x)
    if 'gaussian' in noises:
        x = add_gaussian_noise(x)
    if 'highiso' in noises:
        x = add_high_iso_noise(x)
        #print('high_iso ???')
    if 'salt' in noises:
        x = add_salt_noise(x, multiprocessing)
    if 'wavelet' in noises:
        x = add_wavelet_noise(x, multiprocessing)
    if 'inpainting' in noises:
        x = add_blue_hole(x)
    return np.clip(x.astype('float32')/255, 0, 1)
        
def compare_im(image, lignes=1, n=0):
    orig = image.copy()
    
    datagen = preprocess.ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        preprocessing_function=None,
        validation_split=0.0)
    
    
    datagen.fit([image])
    noisy = add_noise(image)
    forme = image.shape
    reshape = (1, forme[0], forme[1], forme[2])
    predict = np.clip(model.predict_generator(generator=datagen.flow(np.reshape(noisy, reshape), np.reshape(image, reshape)))[0], 0, 1)
    plt.subplot(lignes,3,(3*n+1))
    plt.imshow(noisy)
    plt.axis('off')
    plt.subplot(lignes,3,(3*n+2))
    plt.imshow(predict)
    plt.axis('off')
    plt.subplot(lignes,3,(3*n+3))
    plt.imshow(orig)
    plt.axis('off')
    return predict
    
def sample_images(images):
    for i, image in enumerate(images):
        compare_im(image, len(images), i)
    plt.show()

def predict(image):
    datagen = preprocess.ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        preprocessing_function=add_noise,
        validation_split=0.0)
    
    
    datagen.fit([image])
    forme = image.shape
    reshape = (1, forme[0], forme[1], forme[2])
    return np.clip(model.predict_generator(generator=datagen.flow(np.reshape(image, reshape), np.reshape(image, reshape)))[0], 0, 1)

if dataset == 'cifar100':
    
    y_train = y_train.astype('float32')
    y_train /= 255
    y_test = y_test.astype('float32')
    y_test /= 255
    
    print("Preprocessing data...", end=' ', flush=True)
    if not(data_already_peprocessed):
        x_train, x_test = add_noise(y_train), add_noise(y_test)
        print("Done.", flush=True)
    else:
        print("Skip. (already done)", flush=True)
        
#    x_train = x_train.astype('float32')
#    x_train /= 255
#    x_test = x_test.astype('float32')
#    x_test /= 255
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

#input_dim = x_test.shape[1:]
#output_dim = y_test.shape[1:]
    
def generate_data(directory, batch_size=32, noises=[], target_size=(512,512), class_mode='', resize=False):
    """Replaces Keras' native ImageDataGenerator."""
    i = 0
    file_list = []
    file_preprocessed_list = []
    preprocessed = False
    for f in os.scandir(os.path.join(directory, class_mode)):
        file_list.append(f.path)
    if os.path.isdir(directory + '_'+noises[0]):
        for f in os.scandir(os.path.join(directory + '_'+noises[0], class_mode)):
            file_preprocessed_list.append(f.path)
            preprocessed = True
    if not(preprocessed):
        file_preprocessed_list = [""]*len(file_list)
    files = np.stack((file_list, file_preprocessed_list), axis=1)
    while True:
        image_batch = []
        noisy_batch = []
        for b in range(batch_size):
            if i >= len(files):
                i = 0
                np.random.shuffle(files)
            sample = files[i]
            #print(sample)
            i += 1
        #try:
            #print(sample)
            #image = cv2.resize(cv2.imread(sample[0]), target_size)[:,:,:3] # Remove alpha channel
            image = plt.imread(sample[0], format='rgb')[:, :, :3]
            if resize:
                image = cv2.resize(image, target_size)
            image_batch.append(image.astype(float))
        #except Exception as e:
        #    print("data flow error : {}".format(e))
                
            if preprocessed:
            #try:
                #print(sample)
                #image = cv2.resize(cv2.imread(sample[0]), target_size)[:,:,:3] # Remove alpha channel
                image = plt.imread(sample[1], format='rgb')[:, :, :3]
                if resize:
                    image = cv2.resize(image, target_size)
                noisy_batch.append(image.astype(float))
            #except Exception as e:
            #    print("data flow error : {}".format(e))
        
        #print(noisy_batch.shape, image_batch.shape)
        image_batch = np.array(image_batch)
        if preprocessed:
            #print('preprocessed')
            noisy_batch = np.array(noisy_batch)/255
        else:
            try:
                noisy_batch = add_noise(np.array(image_batch), adapt=False, batch = True)
            except Exception as e:
                print("noisy_batch error :",e,image_batch.shape,sample)
                if len(files) > 0:
                    del(files[i])
                else:
                    raise ValueError('No more good files ! WTF')
                continue
        
        if np.average(noisy_batch[0]) == 1:
            print("Warning ! Average value of noisy images is 1. There might be a problem somewhere")
        #print(np.max(noisy_batch[0]), np.average(noisy_batch[0]))
        
        yield noisy_batch, image_batch/255#, np.ones(len(image_batch))

# Yep, that data generator actually does nothing... it is just used to flow training data
datagen = preprocess.ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False,
    preprocessing_function=None,
    validation_split=0.0)


#folder_flow = datagen.flow_from_directory('/home/aviallon/AI/datasets/DIV2K_train_HR/',
#                                          class_mode='binary',
#                                          batch_size=batch_size,
#                                          target_size=(512, 512))
#
#validation_generator = datagen.flow_from_directory('/home/aviallon/AI/datasets/DIV2K_valid_HR/',
#                                          class_mode='binary',
#                                          batch_size=batch_size,
#                                          target_size=(512, 512))

steps_epoch, val_steps = None, None

if dataset == 'cifar100':
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)
    
    train_generator = datagen.flow(x_train, y_train,batch_size=batch_size)
    validation_generator = (x_test, y_test)

else:
    train_generator = generate_data('./data',
                                    noises=noises,
                                    batch_size=batch_size,
                                    target_size=(512, 512))
    
    #datagen.fit(train_generator)
    
    validation_generator = generate_data('./data_val_small',
                                         noises=noises,
                                         batch_size=batch_size,
                                         target_size=(512, 512))
    
    steps_epoch, val_steps = len(os.listdir('./data')) // batch_size, len(os.listdir('./data_val_small')) // batch_size

model = Sequential()

n_colors = 3
if args.arch == 'simple':
    model.add(Conv2D(20, (5, 5), padding='same', input_shape=(None, None, n_colors)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(n_colors, (5, 5), padding='same'))
    model.add(Activation('relu'))
elif args.arch == 'large':
    model.add(Conv2D(32, (7, 7), padding='same', input_shape=(None, None, n_colors)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(n_colors, (7, 7), padding='same'))
    model.add(Activation('relu'))
elif args.arch == 'large2':
    model.add(Conv2D(32, (7, 7), padding='same', input_shape=(None, None, n_colors)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Conv2DTranspose(n_colors, (7, 7), padding='same'))
    model.add(Activation('relu'))
elif args.arch == 'large3':
    model.add(Conv2D(32, (9, 9), padding='same', input_shape=(None, None, n_colors)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(32, (9, 9), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(n_colors, (7, 7), padding='same'))
    model.add(Activation('relu'))
elif args.arch == 'xlarge':
    model.add(Conv2D(32, (15, 15), padding='same', input_shape=(None, None, n_colors)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(n_colors, (15, 15), padding='same'))
    model.add(Activation('relu'))
elif args.arch == 'heavy':
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(None, None, n_colors)))
    model.add(Conv2D(32, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.05))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Conv2DTranspose(15, (5, 5), padding='same'))
    model.add(Conv2DTranspose(6, (7, 7), padding='same'))
    model.add(Conv2DTranspose(n_colors, (15, 15), padding='same'))
    model.add(Activation('relu'))

def DSSIM_MSE():
    def loss(y_true, y_pred):
        return 0.6*keras.losses.mean_squared_error(y_true,y_pred) + 0.4*keras_contrib.losses.DSSIMObjective(y_true,y_pred)
    return loss

opt = keras.optimizers.Nadam()
model.compile(loss=DSSIM_MSE(), optimizer=opt, metrics=['accuracy', 'mse'])

print(model.summary())

class LossInfo(keras.callbacks.Callback):
    def __init__(self, save_best=False):
        self.save_best = save_best
    
    def on_train_begin(self, logs={}):
        self.losses = [np.inf]
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs={}):
        no_improvement = True
        if logs.get('val_loss') < min(self.losses):
            self.best_epoch = epoch
            print("New best loss : {} ({} lower than previous best)".format(logs.get('val_loss'), min(self.losses)-logs.get('val_loss')))
            if self.save_best:
                model_path = os.path.join(save_dir, model_name+'-checkpoint.h5')
                self.model.save(model_path)
                print("Saved model to {}".format(model_path))
            no_improvement = False
        self.losses.append(logs.get('val_loss'))
        if no_improvement:
            print("Best loss : {} at epoch {} (we are {} epochs further)".format(min(self.losses), self.best_epoch, epoch-self.best_epoch))
            
class ValidationProgress(keras.callbacks.Callback):
    def __init__(self, prog_len = 27):
        self.number_batch = -1
        self.n = prog_len
        
    def on_test_begin(self, logs={}):
        self.current_batch = 0
        print("\nTesting : [",end='',flush=True)
        if self.number_batch > 0:
            print("."*self.n,end='',flush=True)
            print("]",end='', flush=True)
    def on_test_batch_begin(self, batch, logs={}):
        if self.number_batch <= 0:
            print(".",end='',flush=True)
    def on_test_batch_end(self, batch, logs={}):
        import sys
        self.current_batch += 1
        if self.number_batch > 0:
            n_dash = (self.current_batch*self.n)//(self.number_batch)
            print("\rTesting : ["+"="*(n_dash)+">"+"."*(self.n-n_dash)+"] ({}/{})".format(self.current_batch, self.number_batch),end='',flush=True)
        else:
            print('\b ', end="", flush=True) 
            sys.stdout.write('\010')
            print("#",end='',flush=True)
    def on_test_end(self, logs={}):
        if self.number_batch <= 0:
            print("]",flush=True)
            self.number_batch = self.current_batch
        else:
            print('', flush=True)

lossinfo = LossInfo(save_best=True)
validationprog = ValidationProgress()
stop_when_no_improvements = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
#checkpoint = ModelCheckpoint(checkpoint_name, monitor="val_loss", verbose=1, save_best_only=True, period=3)


if resume != 0:
    checkpoint_name = os.path.join(save_dir, model_name+'-checkpoint.h5')
    try:
        if os.path.isfile(checkpoint_name):
            model.load_weights(checkpoint_name)
    except ValueError:
        os.rename(checkpoint_name, checkpoint_name+'.old')
else:
    print('Not resuming previous learn.')

# fits the model on batches with real-time data augmentation:
history = model.fit_generator(train_generator,
                    epochs=epochs,
                    workers=18,
                    use_multiprocessing=True,
                    shuffle=True,
                    max_queue_size=30,
                    steps_per_epoch=steps_epoch,
                    validation_steps=val_steps,
                    callbacks = [stop_when_no_improvements, lossinfo, validationprog],
                    initial_epoch = resume,
                    validation_data = validation_generator)


if args.history:
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
#with open(os.path.join(save_dir, model_name+'.json'), 'w') as f:
#    f.write(model.to_json())
print('Saved trained model at %s ' % model_path)

# Score trained model.
if dataset == 'cifar100':
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    compare_im(y_test[0])
