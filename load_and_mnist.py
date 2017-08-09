import csv
import numpy as np
from os.path import join

BATCH_SIZE = 32
DATA_DIR = ''
TESTF = 'train.csv'
TRAINF = 'test.csv'

def load_data(fname):
    with open(join(DATA_DIR,fname),'rb') as f:
        r = csv.reader(f)
        raw_data = [[x[0],x[1:]] for x in r]
    return zip(*raw_data)

def list2vec(vec_list):
	return [np.array(vec) for vec in veclist]


def load_mnist():
    train_labels,train_data = load_data(TRAINF)
    test_labels,test_data = load_data(TESTF)
    train_data = list2vec(train_data)
    test_data = list2vec(test_data)

    return train_labels,train_data,test_labels,test_data

#TODO
def get_data():
	"""
	converts vectors to tfcompatible tensors
	saves test and validation to tf.records
	"""
	pass
