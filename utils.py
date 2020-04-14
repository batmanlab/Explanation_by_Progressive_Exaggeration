import os
import pickle
import numpy as np
from tqdm import tqdm
import scipy.misc as scm
import pdb
import os
from glob import glob
from collections import namedtuple

def crop_center(img,cropx,cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def read_data_file(file_path, image_dir=''):
    attr_list = {}   
    path = file_path
    file = open(path,'r')    
    n = file.readline()
    n = int(n.split('\n')[0]) #  Number of images    
    attr_line = file.readline()
    attr_names = attr_line.split('\n')[0].split() # attribute name
    for line in file:
        row = line.split('\n')[0].split()
        img_name = os.path.join(image_dir, row.pop(0))
        try:
            row = [float(val) for val in row]
        except:
            print(line)
            img_name = img_name + ' '+row[0]
            row.pop(0)
            row = [float(val) for val in row]            
#    img = img[..., ::-1] # bgr to rgb
        attr_list[img_name] = row       
    file.close()
    return attr_names, attr_list

def load_images_and_labels(imgs_names, image_dir, n_class, attr_list, input_size=128, num_channel=3, do_center_crop=False):
    imgs = np.zeros((imgs_names.shape[0], input_size, input_size, num_channel), dtype=np.float32)
    labels = np.zeros((imgs_names.shape[0], n_class), dtype=np.float32)

    for i, img_name in tqdm(enumerate(imgs_names)):
        img = scm.imread(os.path.join(image_dir, img_name))
        if do_center_crop and input_size == 128:
            img = crop_center(img, 150,150)
        img = scm.imresize(img, [input_size, input_size, num_channel])
        img = np.reshape(img, [input_size,input_size,num_channel])
        img = img / 255.0 
        img = img - 0.5
        img = img * 2.0
        imgs[i] = img
        try:
            labels[i] = attr_list[img_name]
        except:
            print(img_name)
            pdb.set_trace()
    labels[np.where(labels==-1)] = 0
    return imgs, labels

def inverse_image(img): 
    img = (img + 1.) * 127.5
    img[img > 255] = 255.
    img[img < 0] = 0.
    return img.astype(np.uint8)
    
def make3d(img, num_channel, image_size, row, col):
    # img.shape = [row*col, h, w, c]
    # final: [row*h, col*w, c]
    if num_channel > 1:
        img = np.reshape(img, [col,row,image_size,image_size,num_channel]) # [col, row, h, w, c]
    else:
        img = np.reshape(img, [col,row,image_size,image_size]) # [col, row, h, w]
    img = unstack(img, axis=0) # col * [row, h, w, c]
    img = np.concatenate(img, axis=2) # [row, h, col*w, c]
    img = unstack(img, axis=0) # row * [h, col*w, c]
    img = np.concatenate(img, axis=0) # [row*h, col*w, c]
    return img


def unstack(img, axis):
    d =img.shape[axis]
    arr = [np.squeeze(a,axis=axis) for a in np.split(img, d, axis=axis)]
    return arr
    
def save_images(realA, realB, fakeB, cycA, sample_file, image_size=128, num_channel = 3):
    img = np.concatenate((realA[:5,:,:,:],fakeB[:5,:,:,:], cycA[:5,:,:,:], realB[:5,:,:,:],
                          realA[5:,:,:,:],fakeB[5:,:,:,:], cycA[5:,:,:,:], realB[5:,:,:,:]), axis=0)
                          
    img = make3d(img, num_channel=num_channel, image_size=image_size, row=5, col=8)                      
    img = inverse_image(img)
    scm.imsave(sample_file, img)