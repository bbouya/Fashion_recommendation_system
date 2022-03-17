'''
This python file is responsible for the image processing.
'''

from cProfile import label
from operator import ilshift, ipow
from sre_constants import ASSERT
from sre_parse import FLAGS
import cv2
import numpy as np
import pandas as pd
from hyper_parameters import *

shuffle = True  
localization = FLAGS.is_localization
imageNet_mean_pixel = [103.939,116.799,123.68]
global_std = 68.76 

IMG_ROWS = 64
IMG_COLS = 64

def get_image(path, x1, y1, x2,y2):
    '''
    :param path : image path
    :param x1: the upper left and lower 
    :param y1 : the upper left and
    :param x2 et y2.
    : return : a numpy array with dimension [img_row, img_col, img_depth]
    '''

    img = cv2.imread(path)
    if localization is True:
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            img = np.zeros((1,IMG_ROWS,IMG_COLS))
        img = cv2.resize(img,IMG_ROWS,IMG_COLS)
        assert img.shape == (IMG_ROWS,IMG_COLS,3)
    else:
        img = cv2.resize(img,IMG_ROWS,IMG_COLS)
    
    img = img.reshape(1, IMG_ROWS,IMG_COLS,3)

    return img

def load_data_numpy(df):
    '''
    :param df: a pandas dataframe with the image paths and localization coordiante
    : return : the numpy representation of the image
    '''

    num_images = len(df)
    image_path_array = df['image_path'].as_matrix()
    label_array = df['category'].as_matrix()
    x1 = df['x1_modified'].as_matrix().reshape(-1, 1)
    y1 = df['y1_modified'].as_matrix().reshape(-1, 1)
    x2 = df['x2_modified'].as_matrix().reshape(-1, 1)
    y2 = df['y2_modified'].as_matrix().reshape(-1, 1)

    bbox_array = np.array([]).reshape(-1,IMG_ROWS,IMG_COLS,3)
    adjusted_std = 1.0/np.sqrt(IMG_ROWS * IMG_COLS * 3)

    for i in range(num_images):
        img = get_image(image)
