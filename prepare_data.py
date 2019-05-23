#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:41:52 2019

@author: aviallon
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import cv2

from multiprocessing import Pool,Process,cpu_count

save_dir = 'data'

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
        
noise_image = np.clip((plt.imread('../datasets/noise/101HPIMG/HPIM3024.JPG')[:,:,:3]*13), 0, 255)
def _add_high_iso_noise(x):
    noise = cv2.repeat(crop(noise_image, 300, 64), 15, 15)
    if x.shape[0] > noise.shape[0] or x.shape[1] > noise.shape[1]:
        raise Exception('Image too big for HIGH ISO noise')
    
    noise_patch = crop(noise, x.shape[0], min(1, abs(noise.shape[0]//2-x.shape[0])))
    noise_patch = noise_patch.astype('float32') / (2*255)
    x = x[:, :, :3].astype('float32') / (255*2)
    res = np.array(x + noise_patch)
    #print(np.max(res))
    return res

def add_high_iso_noise(x, dummy):
    if len(x.shape) > 3:
        return np.array([_add_high_iso_noise(im) for im in x])
    else:
        return _add_high_iso_noise(x)
    
def convert(file, size, func=crop):
    #print(file)
    path, filename = file[0], file[1]
    print("Processing %s..." % filename, end='')
    try:
        x = plt.imread(path, format='rgb')
        plt.imsave(os.path.join(save_dir, filename), func(x, size))
        print("done !")
    except AttributeError:
        print(" attribute error...")
    except Exception as e:
        print("error : ", e)
    
def convert_dir(directory, size, func=crop):
    n_files = len(os.listdir(directory))
    files = []
    for f in os.scandir(directory):
        files.append([f.path, f.name])
    n = int(cpu_count()*1.5)
    #print(files, directory)
    while len(files) > 0:
        jobs = []
        for i in range(min(n, len(files))):
            p = Process(target=convert, args=(files[i], size, func))
            jobs.append(p)
            p.start()
            
        del(files[0:min(n, len(files))])
        for p in jobs:
            p.join(5)
        
        print('==== [{}/{}] ===='.format(n_files-len(files), n_files))
            
        
parser = argparse.ArgumentParser(description='Prepare data for learning.')
parser.add_argument(metavar='directory', dest='directory', help='specify input dir')
parser.add_argument('--size', dest='size', default=512, type=int, help='set new image size')
parser.add_argument('--dir', dest='dir', default=save_dir, help='set output dir')
parser.add_argument('--func', dest='func', default='crop', choices=['crop', 'iso'], help='set preprocessing function')

args = parser.parse_args()

func = crop
if args.func == 'crop':
    func = crop
elif args.func == 'iso':
    func = add_high_iso_noise

save_dir = args.dir
convert_dir(args.directory, args.size, func)