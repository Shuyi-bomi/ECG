import keras.backend as K
from threshhold import PostProcessing, findthreshhold
from modelbuild import plot_roc, plot_confusion_matrix, multilabel_loss, weighted_binary_crossentropy
from keras.models import Model, load_model
from keras.losses import binary_crossentropy
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import json
from datetime import date, datetime
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
#from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='', type=str, dest='folder')
args = parser.parse_args()


config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
#config.gpu_options.per_process_gpu_memory_fraction = .9
config.gpu_options.allow_growth = True

#sess = tf.compat.v1.InteractiveSession(config=config)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
    
    
def focal_loss_fixed(y_true, y_pred):
    gamma=2.
    alpha=.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    
# Load parameters
params = json.load(open('config.json', 'r'))
fivedictr = json.load(open('5-fold/fivefdtrid.json', 'r'))
#fivedicval = json.load(open('5-fold/fivefdvalid.json', 'r'))  
params['dense_neurons'] = 128

# Load data and label
with open("indextrain.txt", "r") as fp:
    train_id = json.load(fp)
with open("indextest.txt", "r") as fp:
    test_id = json.load(fp)
train_y = np.load('train_y.npy')
test_y = np.load('test_y.npy')



model = []
file_name = os.listdir('result/'+args.folder)
for i in range(5):
    model.append(load_model('result/'+args.folder+file_name[i],custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy}))

    
def plotsave(model, fivedictr, train_id, train_y, test_id, test_y, args, params):
    # Visualize the classification result
    # First load the test set into memory
    '''
    train_x = []
    train_yy = []
    for ID in train_id:
        train_x.append(np.load('ecgtrain/' + str(ID) + '.npy'))
        train_yy.append(train_y[train_y[:,0]==ID][0,1:]) 
    train_x = np.array(train_x)
    train_y = np.array(train_yy)
    '''
    test_x = []
    #test_yy = []
    for ID in test_id:
        test_x.append(np.load('ecgtest/' + str(ID) + '.npy'))
        #test_yy.append(test_y[test_y[:,0]==ID][0,1:]) 

    test_x = np.array(test_x)
    
    #test_y = np.array(test_yy)
    test_pos_predict = np.zeros((test_x.shape[0],27))
    #train_pos_predict = np.zeros((train_x.shape[0],27))
    
    for j in range(5):
        #train_pos_predict = train_pos_predict + model[j].predict(train_x)
        test_pos_predict = test_pos_predict + model[j].predict(test_x)

    #train_pos_predict = train_pos_predict/5
    test_pos_predict = test_pos_predict/5
    #test_predict_onehot = (test_pos_predict >= 0.5).astype(int)
    print('finish predict')
    
    
    score = {}
    steps = [.05]#np.arange(0.01, 0.06, .01)
    
    abbr_list = list(params['class_name'].values())
    
    for k in range(len(steps)):
        thresh_opt = findthreshhold(fivedictr, train_y, params, model, step = steps[k], folder = 'result/'+args.folder )
        thresh_opt = np.array(thresh_opt) #(5, 27)
        thresh_opt = np.mean(thresh_opt,0)
        print(thresh_opt)


        postprocessing = PostProcessing(fold=5, threshold = list(thresh_opt))
        test_predict_onehot = postprocessing.run(test_pos_predict)
        binaryf1,recall,precision = postprocessing.obtain_score(test_y, test_predict_onehot)
        f1 = f1_score(test_y, test_predict_onehot, average='weighted')
        
        score[k] = {}
        score[k]["f1"] = f1
        score[k]["binaryf1"] = binaryf1
        score[k]["recall"] = recall
        score[k]["precision"] = precision
        score[k]['thresh_opt'] = list(thresh_opt)
        
        #plot
        today = datetime.now()#date.today()
        
        # Confusion matrix
        conf_matrix = []
        for i in range(len(abbr_list)):
            conf_matrix.append(confusion_matrix(test_y[:, i], test_predict_onehot[:, i]))
        plt.figure(figsize=(42, 35))
        for i in range(len(abbr_list)):
            plt.subplot(5, 6, i+1)
            plot_confusion_matrix(abbr_list[i], conf_matrix[i])

        plt.tight_layout()
        plt.savefig('result/'+args.folder+'multilabel_conf_' + str(k) + today.strftime("%m%d%H") + '.png')
        print('finish plotting Confusion matrix %1d '% (i))

    with open("result/"+args.folder+"/scoretest"+".json", "w") as fp:
        json.dump(score, fp)
    
    
    # ROC & AUC
    plt.figure(figsize=(24, 20))
    for i in range(len(abbr_list)):
        plt.subplot(5, 6, i+1)
        plot_roc(abbr_list[i], test_y[:, i], test_pos_predict[:, i], 'blue')

    plt.tight_layout()
    plt.savefig('result/'+args.folder+'/multilabel_roc_' + today.strftime("%m%d%H")  +'.png')
    print('finish plot roc')
    


'''
thresh_opt = findthreshhold(fivedictr, train_y, params, model, postprocessing = PostProcessing(fold=5),folder='result/0707/wt1focal/')
thresh_opt = np.array(thresh_opt)
thresh_opt = np.mean(thresh_opt,0)
print(thresh_opt)
test_x = []
for ID in test_id:
    test_x.append(np.load('ecgtest/' + str(ID) + '.npy'))
    #test_yy.append(test_y[test_y[:,0]==ID][0,1:]) 

test_x = np.array(test_x)
test_y = test_y[:,1:]
test_pos_predict = np.zeros((test_x.shape[0],27))

for j in range(5):
    test_pos_predict = test_pos_predict + model[j].predict(test_x)

test_pos_predict = test_pos_predict/5
postprocessing = PostProcessing(fold=5, threshold = list(thresh_opt))
test_predict_onehot = postprocessing.run(test_pos_predict)
binaryf1,recall,precision = postprocessing.obtain_score(test_y, test_predict_onehot)
f1 = f1_score(test_y, test_predict_onehot, average='weighted')
score = {}
score["f1"] = f1
score["binaryf1"] = binaryf1
score["recall"] = recall
score["precision"] = precision
score['thresh_opt'] = list(thresh_opt)


with open("result/0707/wt1focal/"+"/scoretest"+".json", "w") as fp:
    json.dump(score, fp)
    
#plot
abbr_list = [str(i) for i in range(1,28)]


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
'''    

plotsave(model, fivedictr, train_id, train_y, test_id, test_y[:,1:].astype(np.float32), args, params)