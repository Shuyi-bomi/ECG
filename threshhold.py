from modelbuild import DataGenerator
from keras.utils import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json
import time
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
#from tqdm import tqdm_notebook
#from keras_tqdm import TQDMNotebookCallback
#from keras_radam import RAdam
import tensorflow_addons as tfa




        
def compute(labels, outputs):
    #y_true: labels, y_pred: outputs
    f1 = f1_score(labels, outputs)

    return f1  



class PostProcessing():


    def __init__(self,fold, step = .1, threshold = None):
        #threshold should be list
        self.fold = fold
        if not threshold:
            self.threshold = .5#float(open(f"threshold_{self.fold}.txt", "r").read())#0.5#0.1
        else:
            self.threshold = threshold
        self.metric = compute
        self.step = step


    def run(self,predictions):

        predictions_processed = predictions.copy()
        for i in range(27):
            predictions_processed[np.where(predictions_processed[:,i] >= self.threshold[i]), i] = 1
            predictions_processed[np.where(predictions_processed[:,i] < self.threshold[i]), i] = 0
        

        return predictions_processed

    def find_opt_thresold(self, labels, outputs):
        threshold_grid = np.arange(0.01, 0.99, self.step).tolist()
        threshold_opt = []
        start = time.time()
        for i in range(27):
            unit_threshold= partial(self._unit_threshold,labels=labels[:,i],outputs=outputs[:,i])
            with ProcessPoolExecutor(max_workers=20) as pool:
                result = pool.map(
                     unit_threshold, threshold_grid
            )
            scores = np.array(list(result))
            a = np.where(scores == np.max(scores))
            threshold_opt.append(threshold_grid[a[0][0]] ) 
        print(f'Processing time: {(time.time() - start)/60}')
        
        return threshold_opt

    def _unit_threshold(self,threshold,labels,outputs):

        predictions = outputs.copy()

        predictions[np.where(predictions >= threshold)] = 1
        predictions[np.where(predictions < threshold)] = 0

        return self.metric(labels, predictions)

    def update_threshold(self, threshold, folder='./'):
        f = open(f"{folder}threshold_{self.fold}.txt", "a")
        f.write(str(threshold)+"\n")
        f.close()
        self.threshold = threshold
        
    def obtain_score(self, labels, outputs):
        f1 = []
        recall = []
        precision = []
        for i in range(27):
            f1.append(f1_score(labels[:,i], outputs[:,i]))
            recall.append(compute(labels[:,i], outputs[:,i]) )
            precision.append(precision_score(labels[:,i], outputs[:,i]) )
            #recall.append(recall_score(labels[:,i], outputs[:,i]))
        return f1,recall,precision

    
#postprocessing = PostProcessing(fold=5)
def findthreshhold(fivedictr, train_y, params, model, step=.1, folder='/'):
    threshold_opt = []
    score = {}
    postprocessing = PostProcessing(fold=5, step = step)
    for j in range(5):
        dg = DataGenerator(fivedictr[str(j)], train_y, batch_size=params["batch_size"], 
                           n_classes=params["disease_num"], shuffle=False)
        train_pos_predict = model[j].predict(dg, verbose=1)
        if j>=1:
            train_pos_predict = np.vstack((train_pos_predict, 
                              model[j].predict((np.load('ecgtrain/' + str(fivedictr[str(j)][-1]) + '.npy')).reshape((1,5000,12))))) 
        

        trainynow = []
        for ID in fivedictr[str(j)]:
            trainynow.append(train_y[train_y[:,0]==ID][0,1:]) 
        trainynow = np.array(trainynow)
        print(train_pos_predict.shape, trainynow.shape)
        #train_yy.append(trainynow)

        threshold = postprocessing.find_opt_thresold(trainynow, train_pos_predict) #should be train
        threshold_opt.append(threshold)
        postprocessing.update_threshold(threshold, folder)
        train_predict_onehot = postprocessing.run(train_pos_predict)
        f1,recall,precision = postprocessing.obtain_score(trainynow, train_predict_onehot)
        
        score[j] = {}
        score[j]["f1"] = f1
        score[j]["recall"] = recall
        score[j]["precision"] = precision
        
        print(f1,recall,precision)
    with open(folder+"/score"+".json", "w") as fp:
        json.dump(score, fp)
    
    return threshold_opt
    
    
    
    