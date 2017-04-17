'''
 This script is intended to create csv files, namely
 train.csv
 test.csv
 from all the xml files from the opensubtitle dataset.

 This is intended to convert it to the same format than the UDC dataset.
'''
import numpy as np
from random import sample
import random
from lxml import etree
from os import listdir
import glob
import sys
import os
import pandas as pd
import math
import tensorflow as tf

dir_xml_path = '/home/bbaga/Projects/OpenSubBot/Data/Part'

file_train = '../data/train.csv'
file_test = '../data/test.csv'
file_valid = '../data/valid.csv'

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist

limit = {
        'maxqa' : 25,
        'minqa' : 2,
        }
    
def split_dataset(x, y, ratio = [0.7, 0.15, 0.15]):
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]
    
    return (trainX,trainY), (testX,testY), (validX,validY)

def filter_sentence(s, whitelist):
    # format the sentence according to the whitelist
    s = s.lower()
    s = "".join(c for c in s if c in whitelist).strip()
    nb_w = len(s.split(' '))
    if s == "" or nb_w < limit['minqa'] or nb_w > limit['maxqa']:
        return None
    return s

def read_xml(data_path):
    with tf.gfile.GFile(data_path, 'r') as f:
        text = f.read()
    root = etree.fromstring(text)
    lst = []
    for s in root.xpath('//document/s'):
        ws = [w.text for w in s.findall('w')]
        sentence = " ".join(ws)
        sentence = filter_sentence(sentence, EN_WHITELIST)
        if sentence is not None:
            lst.append(sentence)
    return lst

def add_xml_data(lst):
    if len(lst)%2 !=0: #make sure the number of lines is even
        lst = lst[:-1]
    # separate the "questions" and "answers by assuming the files are always alternating
    x = lst[::2]
    y = lst[1::2]
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = split_dataset(x,y)
    
    # now we format the data so it fits the UDC dataset for training testing and validating
    
    # TRAINING
    #  set half of Utterance on purpose wrong (Label 0)
    idx = sample(list(np.arange(len(x_train))), len(x_train))
    x_train = [ x_train[i] for i in idx]
    y_train = [ y_train[i] for i in idx]
    rate_label_1 = 0.5
    k = (int)(len(x_train)*rate_label_1)
    # add rate_label_1 in the file
    label_train = list(np.zeros(len(x_train)-k)) + list(np.ones(k))
    # shuffle the label 0
    idx_k = list(range(0,k))
    random.shuffle(idx_k)
    y_train = [ y_train[i] for i in idx_k] + y_train[k:]
    for i in range(0,len(x_train)):
        f_train.write(x_train[i] + ',' + y_train[i] + ',' + str((int)(label_train[i]))+"\n")
    
    # TESTING
    test_len = len(x_test)
    for i in range(0,len(x_test)):
        distr = ''
        for j in range(0,9):
            distr += ',' + y_test[(j+1)%test_len]
        f_test.write(x_test[i] + ',' + y_test[i] + distr+"\n")

    # VALIDATION
    valid_len = len(x_valid)
    for i in range(0, len(x_valid)):
        distr = ''
        for j in range(0,9):
            distr += ',' + y_valid[(j+1)%valid_len]
        f_valid.write(x_valid[i] + ',' + y_valid[i] + distr+"\n")


# recursively read all xml given a directory path or a list of directory
def process_xmls(dir_list):
    print('Processing ' + dir_list + ' ...')
    for filename in glob.glob(dir_list+'/**/*.xml'):
        current = read_xml(filename)
        add_xml_data(current)


#SCRIPT RUN

# initialization
f_train = tf.gfile.GFile(file_train, 'w')
f_test = tf.gfile.GFile(file_test, 'w')
f_valid = tf.gfile.GFile(file_valid, 'w')

process_xmls(dir_xml_path)

# closure
f_train.close()
f_test.close()
f_valid.close()

