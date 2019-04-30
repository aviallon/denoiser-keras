import os
import time
import argparse

import plaidml.keras
plaidml.keras.install_backend()

import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
#from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, LeakyReLU
#from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
import keras.preprocessing.image as preprocess

np.random.seed()

model_name = 'electronic_noise_denoiser.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(save_dir, model_name)

parser = argparse.ArgumentParser(description='Denoise a photo.')
parser.add_argument('-i', '--input', metavar='INPUT_FILE',
                    help='input image')
'''parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')'''
parser.add_argument('--model', help='specify model file')
parser.add_argument('--output', help='output file')

args = parser.parse_args()
print(args)
img_path = args.input
if args.model != None:
    model_path = args.model
output_filename = 'denoised_'+img_path
if type(args.output) !=  None:
    output_filename = args.output
    
print("Using model : {}".format(model_path))

model = load_model(model_path)

def load_image(path):
    temp = plt.imread(path, format='rgb')
    if temp.shape[2] > 3:
        temp = temp[:, :, 0:3]
    return temp/255

def predict(image):
    datagen = preprocess.ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        preprocessing_function=None,
        validation_split=0.0)
        
    datagen.fit([image])
    forme = image.shape
    reshape = (1, forme[0], forme[1], forme[2])
    return np.clip(model.predict_generator(generator=datagen.flow(np.reshape(image, reshape), np.reshape(image, reshape)))[0], 0, 1)

#plt.imsave(predict(load_image(img_path)), 'denoised_'+img_path)

plt.imsave(output_filename, predict(load_image(img_path)))