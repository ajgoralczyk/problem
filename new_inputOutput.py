import os, struct
import datetime
from array import array as simple_array
from numpy import *

def readMnist(dataset = "training", path = "."):
    if dataset is "training":
        filename_img = os.path.join(path, 'train-images-idx3-ubyte')
        filename_label = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        filename_img = os.path.join(path, 't10k-images-idx3-ubyte')
        filename_label = os.path.join(path, 't10k-labels-idx1-ubyte')

    file_label = open(filename_label, 'rb')
    magic_nr, size = struct.unpack(">II", file_label.read(8))
    label = simple_array("b", file_label.read())
    file_label.close()

    file_img = open(filename_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", file_img.read(16))
    img = simple_array("B", file_img.read())
    file_img.close()

    index = [ k for k in xrange(size) ] # if label[k] in digits
    images = zeros((len(index), rows, cols), dtype=uint8)
    labels = zeros((len(index), 1), dtype=int8)
    for i in xrange(len(index)):
        images[i] = array(img[ index[i]*rows*cols : (index[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = label[index[i]]
    
    return images, labels

def printDataToFile(id, ab, bestx):
    output_file = open('results_'+ str(id) + '.txt', 'a') # append
    output_file.write(ab  + '\n')
    for i in xrange(len(bestx)):
        output_file.write(str(bestx[i].fitness) + '\n')
    output_file.close()

def printTime(id, time, text=''):
    output_file = open('results_'+ str(id) + '.txt', 'a')
    output_file.write(text + str(time) + '\n')
    output_file.close()

def printText(text):
    output_file = open('my_log.txt', 'a')
    output_file.write(text + '\n')
    output_file.close()










