import gzip
import struct

import numpy

path = './mnist/{}-{}-idx{}-ubyte.gz'

def __read(name, type):
    file = path.format(name, type, '1' if type == 'labels' else '3')
    def read_label():
        with gzip.open(file, 'rb') as f:
            magic, num = struct.unpack('>2I', f.read(8))
            return numpy.frombuffer(f.read(), dtype=numpy.uint8)
    def read_image():
        with gzip.open(file, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>4I', f.read(16))
            return numpy.frombuffer(f.read(), dtype=numpy.uint8).reshape(num, rows*cols)
    return eval('read_{}'.format(type[:-1]))


read_train_images = __read('train', 'images')
read_train_labels = __read('train', 'labels')
read_test_images = __read('t10k', 'images')
read_test_labels = __read('t10k', 'labels')


def get_num_image(num, func = read_train_images, label_func = read_train_labels):
    array = func()
    labels = label_func()
    print(len(labels))
    for i in range(len(labels)):
        if(labels[i] == num):
            print(labels[i])
            yield array[i].reshape((28, 28))