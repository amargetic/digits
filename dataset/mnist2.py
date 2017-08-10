import csv
import numpy as np
import os
from os.path import join
import tensorflow as tf
from PIL import Image
from StringIO import StringIO

DATA_DIR = '/Users/RyanBerger/digit_project/digits/dataset/mnist'
TESTF = 'train.csv'
TRAINF = 'test.csv'

def load_data(fname):
    with open(join(DATA_DIR,fname),'rb') as f:
        r = csv.reader(f)
        raw_data = [[x[0],x[1:]] for x in r]
    return raw_data

def load_mnist():
    training = load_data(TRAINF)
    testing = load_data(TESTF)
    return [('train',training),('test',testing)]

def create_jpegs(data):
    splt,lst = data
    labels,images = zip(*lst)
    splt_path = join(DATA)
    for idx,img in enumerate(lst):
        label = labels[idx]
        tmp_path = join(splt_path,label)
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)
        tmp_path = join(splt_path,'img_%d.jpg'%idx)
        img = np.asarray(img, dtype=np.float32)
        img = np.reshape(img,(28,28))
        img = img*255
        im = Image.fromarray(img).convert('RGB')
        im.save(tmp_path)



