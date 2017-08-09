import csv
import numpy as np
from os.path import join
import tensorflow as tf

#TODO REPLACE WITH FLAGS
BATCH_SIZE = 32
DATA_DIR = '/Users/RyanBerger/digit_project/digits/dataset/mnist'
TESTF = 'train.csv'
TRAINF = 'test.csv'

def load_data(fname):
    with open(join(DATA_DIR,fname),'rb') as f:
        r = csv.reader(f)
        raw_data = [[x[0],x[1:]] for x in r]
    return zip(*raw_data)

def list2vec(vec_list):
	return np.array(np.array(vec) for vec in vec_list)


def load_mnist():
    train_labels,train_data = load_data(TRAINF)
    test_labels,test_data = load_data(TESTF)
    train_data = list2vec(train_data)
    test_data = list2vec(test_data)

    dataset = [[train_data,train_labels,'train'], \
                [test_data,test_labels,'test']]

    return dataset

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images, labels, name):

    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(DATA_DIR, name + '.tfrecords')
    
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(len(images)):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def get_data():
    """
    converts vectors to tfcompatible tensors
    saves test and validation to tf.records
    """
    dataset = load_mnist()

    for data in dataset:
        images,labels,name = data
        convert_to_tfrecord(images, labels, name)













