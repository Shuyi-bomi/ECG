#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 05:49:18 2021

@author: apple
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import keras.backend as K
from keras import backend 
#from tqdm import tqdm_notebook
#from keras_tqdm import TQDMNotebookCallback
#from keras_radam import RAdam
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,classification_report
from modelbuild import model_build, model_train, model_save, model_load, plot_roc, plot_confusion_matrix
from threshhold import findthreshhold
#from ecg_preprocessing import val_split, load_cardiologist_test_set,multiclass_val_split,all_split
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem, t

import os
import json
from collections import Counter
from datetime import datetime, date
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import time
import argparse


def parse():
    
    parser = argparse.ArgumentParser()
    """
    Hyper-parameter for CNNResnet
    """
    parser.add_argument('--learning_rate', default=0.0001, type=float, dest='learning_rate', help='base learning rate for optimizer')
    parser.add_argument('--dropout', default=.6, type=float, dest='dropout')
    parser.add_argument('--index', default=1, type=int, dest='index')
      
    return parser.parse_args()

def train(args, params, up=True):
    config = tf.compat.v1.ConfigProto()
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    #config.gpu_options.per_process_gpu_memory_fraction = .9
    config.gpu_options.allow_growth = True
    
    #sess = tf.compat.v1.InteractiveSession(config=config)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    #set_session(tf.compat.v1.Session(config=config)) 

    tf.test.is_gpu_available()
    seed = 2021
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)



    # Load parameters
    
    fivedictr = json.load(open('5-fold/fivefdtrid.json', 'r'))
    fivedicval = json.load(open('5-fold/fivefdvalid.json', 'r'))  
    params['dense_neurons'] = 128
    params['multiply'] = args.multiply
    params['dropout'] = args.dropout
    params['learning_rate'] = args.learning_rate
    params['if_upsamp'] = args.if_upsamp
    # Load data and label
    if params['if_upsamp']:
        with open("indextrainup.txt", "r") as fp:
            train_id = json.load(fp)
    else:
        with open("indextrain.txt", "r") as fp:
            train_id = json.load(fp)
    with open("indextest.txt", "r") as fp:
        test_id = json.load(fp)
    train_y = np.load('train_yup2.npy') if params['if_upsamp'] else np.load('train_y.npy')
    test_y = np.load('test_y.npy') 

    #if K.backend()=='tensorflow':
    #    backend.set_image_data_format("channels_last")

    # build model
    # model, parallel_model = model_build(params)

    # model training (we here train the model for 10 times to calculate the mean and CIs)
    model = []
    for j in (range(5)):

        modelnow = model_build(params)#CnnResNet(params)
        modelnow = model_train(modelnow, fivedictr[str(j)], train_y, fivedicval[str(j)], train_y, params)
        modelnow.save('result/'+ args.folder +'multilabel_model_' + (datetime.today()).strftime("%m%d%H") + str(j) + '.h5') #save model
        model.append(modelnow)
        #time.sleep(18)
        
    test_x = []
    for ID in test_id:
        test_x.append(np.load('ecgtest/' + str(ID) + '.npy'))
        #test_yy.append(test_y[test_y[:,0]==ID][0,1:]) 

    test_x = np.array(test_x)
    
    test_pos_predict = np.zeros((test_x.shape[0],27))

    for j in range(5):
        test_pos_predict = test_pos_predict + model[j].predict(test_x)

    test_pos_predict = test_pos_predict/5
    test_predict_onehot = (test_pos_predict >= 0.5).astype(int)
    
    
    abbr_list = list(params['class_name'].values())
    
    today = date.today()
    # ROC & AUC
    plt.figure(figsize=(24, 20))
    for i in range(len(abbr_list)):
        plt.subplot(5, 6, i+1)
        plot_roc(abbr_list[i], test_y[:, i], test_pos_predict[:, i], 'blue')

    plt.tight_layout()
    plt.savefig('multilabel_roc_' + today.strftime("%m%d") + '.png')
    print('finish plot roc')
    # Confusion matrix
    conf_matrix = []
    for i in range(len(abbr_list)):
        conf_matrix.append(confusion_matrix(test_y[:, i], test_predict_onehot[:, i]))
    plt.figure(figsize=(42, 35))
    for i in range(len(abbr_list)):
        plt.subplot(5, 6, i+1)
        plot_confusion_matrix(abbr_list[i], conf_matrix[i])

    plt.tight_layout()
    plt.savefig('multilabel_conf_' + today.strftime("%m%d") + '.png')
    print('finish plotting')
    
    
    return model, train_id, train_y, test_id, test_y

def plotsave(model,train_id, train_y, test_id, test_y, class_name):
    test_x = []
    for ID in test_id:
        test_x.append(np.load('ecgtest/' + str(ID) + '.npy'))
        #test_yy.append(test_y[test_y[:,0]==ID][0,1:]) 

    test_x = np.array(test_x)
    
    test_pos_predict = np.zeros((test_x.shape[0],27))

    for j in range(5):
        test_pos_predict = test_pos_predict + model[j].predict(test_x)

    test_pos_predict = test_pos_predict/5
    test_predict_onehot = (test_pos_predict >= 0.5).astype(int)
    
    abbr_list = class_name


    today = date.today()
    # ROC & AUC
    plt.figure(figsize=(24, 20))
    for i in range(len(abbr_list)):
        plt.subplot(5, 6, i+1)
        plot_roc(abbr_list[i], test_y[:, i], test_pos_predict[:, i], 'blue')

    plt.tight_layout()
    plt.savefig('multilabel_roc_' + today.strftime("%m%d") + '.png')
    print('finish plot roc')
    # Confusion matrix
    conf_matrix = []
    for i in range(len(abbr_list)):
        conf_matrix.append(confusion_matrix(test_y[:, i], test_predict_onehot[:, i]))
    plt.figure(figsize=(42, 35))
    for i in range(len(abbr_list)):
        plt.subplot(5, 6, i+1)
        plot_confusion_matrix(abbr_list[i], conf_matrix[i])

    plt.tight_layout()
    plt.savefig('multilabel_conf_' + today.strftime("%m%d") + '.png')
    print('finish plotting')


parser = argparse.ArgumentParser()
"""
Hyper-parameter for CNNResnet
"""
parser.add_argument('--learning_rate', default=0.0001, type=float, dest='learning_rate', help='base learning rate for optimizer')
parser.add_argument('--dropout', default=.6, type=float, dest='dropout')
parser.add_argument('--multiply', default=2, type=int, dest='multiply')
parser.add_argument('--folder', default='', type=str, dest='folder')
parser.add_argument('--if_upsamp', default=0, type=int, dest='if_upsamp')


args = parser.parse_args()
params = json.load(open('config.json', 'r'))
model,train_id, train_y, test_id, test_y = train(args, params)
#plotsave(model,train_id, train_y, test_id, test_y[:,1:],list(params['class_name'].values()))