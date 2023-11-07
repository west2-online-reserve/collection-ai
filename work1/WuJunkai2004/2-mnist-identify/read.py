import gzip
import struct

import numpy

path = './mnist/{}-{}-idx3-ubyte.gz'

def __read(name, type):
    file = path.format(name, type)
    def read():
        with gzip.open(file, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>4I', f.read(16))
            return numpy.frombuffer(f.read(), dtype=numpy.uint8).reshape(num, rows*cols)
    return read

read_train_images = __read('train', 'images')
read_train_labels = __read('train', 'labels')
read_test_images = __read('t10k', 'images')
read_test_labels = __read('t10k', 'labels')