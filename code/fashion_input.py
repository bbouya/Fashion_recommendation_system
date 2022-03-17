'''
This python file is responsible for the image processing.
'''

from operator import ilshift, ipow
from sre_parse import FLAGS
import cv2
import numpy as np
import pandas as pd
from hyper_parameters import *

shuffle = True  
localization = FLAGS.is_localization
imageNet_mean_pixel = [103.939,116.799,123.68]