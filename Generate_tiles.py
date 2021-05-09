#!/usr/bin/env python
# coding: utf-8

# ### 0] Import libs

# In[1]:


import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
os.environ['PATH'] = r"openslide-win64-20171122\bin" + ";" + os.environ['PATH']
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET
parser = ET.XMLParser(encoding="utf-8")
import cv2 as cv
import scipy.ndimage
import xmltodict, json
import pandas as pd
import time


# In[2]:

path_data = r'D:\Users\Julien\Documents\ANTOINE\Notebooks\DATA\\'
images_folder = os.path.join(path_data,'images\\')
annotations_folder = os.path.join(path_data,'annotations\\')


# ### 1] Definition of a function to get the tiles fro mthe generator

# In[3]:


def compute_max_addresses(DZG,tile_size,level,overlap):
    """
    input:
    - Tile generator DZG
    - The size of the tile
    - the level of observation
    - the value of overlap
    
    output:
    - the max value of the adresses for a tile in  the slide
    """
    lvl_dim = DZG.level_dimensions
    #size of the whole slide image with level k

    new_w, new_h = lvl_dim[level]
    address_max_w, address_max_h = (np.array([new_w, new_h])/tile_size).astype('int') - overlap


    #max value of addresses
    return(address_max_w,address_max_h)


# In[4]:


def get_tile_1(DZG, level, address_w,address_h):
    """
    input:
    - Tile generator DZG
    - level of observation
    - adress width of the tile
    - adress heigh of the tile
    output:
    - the image tile 
    """
    ###Choose level 
    lvl_count = DZG.level_count
    
    print('the max level is : {}'.format(lvl_count))    
    if level >= lvl_count:
        print('the level count is too high')
    else:
        lvl_dim = DZG.level_dimensions
        print('the size of the whole slide image is: {}'.format(lvl_dim[level]))

        tile = DZG.get_tile(level,address = np.array([address_w,address_h]))
        img = tile
        return img


# In[1]:


def annotation_to_dataframe(annotation_number,filename):
    
    """
    input:
    - the number of the annotation (written in the xml)
    - the filename (ex: tumor_110)
    
    output:
    'dataframe with 3 columns:
    1_ the order of the vertex
    2_ the value of the X coordinate of the vertex
    3_ the value of the Y coordinate of the vertex
    
    The values of X and Y are the values in the WSI
    
    """

    with open(os.path.join(annotations_folder,filename)+'.tif.xml') as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
    nodes = data_dict['ASAP_Annotations']['Annotations']['Annotation'][annotation_number]['Coordinates']['Coordinate']
    length = len(nodes)
    coord = np.zeros((length,3))
    for i in range(length):
        iter_ = nodes[i]
        coord[i] = np.array([iter_['@Order'], iter_['@X'], iter_['@Y']])
    df = pd.DataFrame(data=coord, columns=['Order', "X",'Y'])
    return df

