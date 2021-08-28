import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPooling1D, Conv1D, Input, add, BatchNormalization, AveragePooling1D, LeakyReLU, multiply, GlobalAveragePooling1D, Reshape, Permute, Concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback
from keras.utils import Sequence, to_categorical

from keras.losses import binary_crossentropy
from keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import json
from collections import Counter
from datetime import date
import tensorflow as tf
import tensorflow.keras.backend as tkb
#from keras.backend.tensorflow_backend import set_session
from tensorflow.compat.v1.keras.backend import set_session
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
#from tqdm import tqdm_notebook
#from keras_tqdm import TQDMNotebookCallback
#from keras_radam import RAdam
import tensorflow_addons as tfa

#subclass to build model
class ResnetidentityBlock(Model):

    def __init__(self, n_filters, index, filter_size=15, strides=1, conv_init="he_normal"):
        super().__init__()


        self.conv1 = Conv1D(n_filters, filter_size, strides=1, padding='same', 
                            name='identity_block' + str(index) + '_' + 'conv_1', 
                            kernel_initializer=conv_init)
        self.bn1 = BatchNormalization(name='identity_block' + str(index) + '_' + 'BN_1')
        self.ac1 = Activation('relu', name='identity_block' + str(index) + '_' + 'relu_1')
        self.ac2 = Activation('relu', name='identity_block' + str(index) + '_' + 'relu_2')
        

    def call(self, inputs, training=None):
        
        residual = inputs        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.ac1(x)
        x = x + residual
        x = self.ac2(x)
        
        return x

    
class ResnetconvBlock(Model):

    def __init__(self, n_filters, index, filter_size=15, conv_init="he_normal"):
        super().__init__()


        self.conv1 = Conv1D(n_filters, filter_size, strides=2, padding='same', 
                            name='conv_block' + str(index) + '_' + 'conv_1', 
                            kernel_initializer=conv_init)
        self.bn1 = BatchNormalization(name='conv_block' + str(index) + '_' + 'BN_1')
        self.ac1 = Activation('relu', name='conv_block' + str(index) + '_' + 'relu_1')
        self.conv2 = Conv1D(n_filters, filter_size, strides=1, padding='same', 
                            name='conv_block' + str(index) + '_' + 'conv_2', 
                            kernel_initializer=conv_init)
        self.bn2 = BatchNormalization(name='conv_block' + str(index) + '_' + 'BN_2')
        self.ac2 = Activation('relu', name='conv_block' + str(index) + '_' + 'relu_2')
        
        self.shortcut = Conv1D(n_filters, filter_size, strides=2, padding='same',
                         name='conv_block' + str(index) + '_' + 'shortcut_conv', 
                          kernel_initializer=params["conv_init"])
        self.bn3 = BatchNormalization(name='conv_block' + str(index) + '_' + 'shortcut_BN')
        self.ac3 = Activation('relu', name='conv_block' + str(index) + '_' + 'relu_3')
        

    def call(self, inputs, training=None):
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        residual = self.shortcut(inputs)
        residual = self.bn3(residual)
        x = x + residual
        x = self.ac3(x)
        
        return x
    

class CnnResNet(tf.keras.Model):

    def __init__(self, params):
        super(CnnResNet, self).__init__()
        #self.input = Input(shape=(5000, 12), name='input')
        self.params = params
        self.conv1 = Conv1D(filters=self.params["conv_num_filters"][0], kernel_size=15, 
               strides=2, padding='same', kernel_initializer=self.params["conv_init"], name='conv_2')
        self.bn1 = BatchNormalization(name='BN_2')
        self.ac1 = Activation('relu', name='relu_2')
        self.mp1 = MaxPooling1D(name='max_pooling_1')
        self.blocks = Sequential(name='dynamic-blocks')
        # 4 blocks here
        for i in range(4):             
            self.blocks.add(ResnetconvBlock(n_filters=self.params["conv_num_filters"][i], index=i+1))
            self.blocks.add(MaxPooling1D(name='max_pooling_' + str(i + 2)) )
            self.blocks.add(ResnetidentityBlock(n_filters=self.params["conv_num_filters"][i], index=i + 1))
        self.ap = AveragePooling1D(name='average_pooling')
        self.fla = Flatten(name='flatten')
        self.fc1 = Dense(self.params["dense_neurons"], kernel_regularizer=l2(self.params["l2"]), name='FC1')
        self.ac2 = Activation('relu', name='relu_3')
        self.dropout = Dropout(rate=self.params["dropout"])
        self.fc2 = Dense(self.params["dense_neurons"], kernel_regularizer=l2(self.params["l2"]), name='FC2')
        self.ac3 = Activation('relu', name='relu_4')
        self.dropout2 = Dropout(rate=self.params["dropout"])
        self.fc3 = Dense(self.params["disease_num"], activation='sigmoid', name='output')
            
        
        
    def call(self, inputs):
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.mp1(x)
        x = self.blocks(x)
        x = self.ap(x)
        x = self.fla(x)
        x = self.fc1(x)
        x = self.ac2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ac3(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return (x)
    
    
    
# Functional API to build model    
def model_build(params):
    def squeeze_excite_block(input_data, ratio=16):
        ''' Create a channel-wise squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
        Returns: a keras tensor
        References
        -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
        '''
        init = input_data
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = init.shape[channel_axis]
        se_shape = (1, filters)

        se = GlobalAveragePooling1D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        if K.image_data_format() == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = multiply([init, se])
        return x
    
    def conv_block(input_data, n_filters, filter_size, index):
        x = Conv1D(n_filters, filter_size, strides=2, padding='same', 
                   name='conv_block' + str(index) + '_' + 'conv_1', 
                   kernel_initializer=params["conv_init"])(input_data)
        x = BatchNormalization(name='conv_block' + str(index) + '_' + 'BN_1')(x)
        x = Activation('relu', name='conv_block' + str(index) + '_' + 'relu_1')(x)
        x = Dropout(rate=.2)(x)
        x = Conv1D(n_filters, filter_size, strides=1, padding='same', 
                   name='conv_block' + str(index) + '_' + 'conv_2',
                   kernel_initializer=params["conv_init"])(x)
        x = BatchNormalization(name='conv_block' + str(index) + '_' + 'BN_2')(x)
        x = Activation('relu', name='conv_block' + str(index) + '_' + 'relu_2')(x)


        shortcut = Conv1D(n_filters, filter_size, strides=2, padding='same',
                         name='conv_block' + str(index) + '_' + 'shortcut_conv', 
                          kernel_initializer=params["conv_init"])(input_data)
        shortcut = BatchNormalization(name='conv_block' + str(index) + '_' + 'shortcut_BN')(shortcut)
        x = squeeze_excite_block(x,ratio=16)
        x = add([x, shortcut], name='conv_block' + str(index) + '_' + 'add')
        x = Activation('relu', name='conv_block' + str(index) + '_' + 'relu_3')(x)
        
        return x

    def identity_block(input_data, n_filters, filter_size, index):
        x = Conv1D(n_filters, filter_size, strides=1, padding='same', 
                   name='identity_block' + str(index) + '_' + 'conv_1', 
                   kernel_initializer=params["conv_init"])(input_data)
        x = BatchNormalization(name='identity_block' + str(index) + '_' + 'BN_1')(x)
        x = Activation('relu', name='identity_block' + str(index) + '_' + 'relu_1')(x)
        x = squeeze_excite_block(x,ratio=16)
        x = add([x, input_data], name='identity_block' + str(index) + '_' + 'add')
        x = Activation('relu', name='identity_block' + str(index) + '_' + 'relu_2')(x)
        
        return x
  
    input_ecg = Input(shape=(5000, 12), name='input')
    x = Conv1D(filters=params["conv_num_filters"][0], kernel_size=15, 
               strides=2, padding='same', kernel_initializer=params["conv_init"], name='conv_2')(input_ecg)
    x = BatchNormalization(name='BN_2')(x)
    x = Activation('relu', name='relu_2')(x)
    x = MaxPooling1D(name='max_pooling_1')(x)
    

    for i in range(4):
        x = conv_block(x, n_filters=params["conv_num_filters"][i], filter_size=params["conv_filter_size"], index=i + 1)
        x = MaxPooling1D(name='max_pooling_' + str(i + 2))(x)
        x = identity_block(x, n_filters=params["conv_num_filters"][i], 
                               filter_size=params["conv_filter_size"], index=i + 1)
            
    #x = AveragePooling1D(name='average_pooling')(x)
    x = Flatten(name='flatten')(x)
    #input_other = Input(shape=(2,), name='input_other') #multiple inputs sex+age
    #x = Concatenate(axis=-1)([x, input_other])
    x = Dense(params["dense_neurons"], kernel_regularizer=l2(params["l2"]), name='FC1')(x)
    x = Activation('relu', name='relu_3')(x)
    #x = Dropout(rate=params["dropout"])(x)
    #x = Dense(params["dense_neurons"], kernel_regularizer=l2(params["l2"]), name='FC2')(x)
    #x = Activation('relu', name='relu_4')(x)
    #x = Dropout(rate=params["dropout"])(x)
    x = Dense(params["disease_num"], activation='sigmoid', name='output')(x)
    
    model = Model(inputs=input_ecg, outputs=x) #[input_ecg,input_other]
    
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
        #parallel_model = multi_gpu_model(model, params["gpu"])
        #parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #parallel_model = multi_gpu_model(model, params["gpu"])
    
    return model

def get_class_weights(total_counts, class_positive_counts, multiply):
    """
    Calculate class_weight used in training
    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean 
    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights

class CompLoss():
    def __init__(self, weights=None):
        super().__init__()

        self.weights = weights

    def forward(self, y_pred, y_true):

        if self.weights != None:
            binary_crossentropy(y_true, y_pred)
        else:

            tp = tf.math.reduce_mean(y_true * y_pred, dim=0)
            tn = tf.math.reduce_mean((1 - y_true) * (1 - y_pred), dim=0)
            fp = tf.math.reduce_mean((1 - y_true) * y_pred, dim=0)
            fn = tf.math.reduce_mean(y_true * (1 - y_pred), dim=0)

            accuracy = (tp+tn)/(fp+fn+tp+tn)
            accuracy = torch.mean(accuracy)

        return 1- accuracy
    
def weighted_binary_crossentropy(y_true, y_pred):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tf.convert_to_tensor(tkb.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))
    pos_wts =[9.481893394818934,7.078080903104421,79.83995815899581,124.05339805825243,32.80708661417323,
              14.416517055655296,10.748707813925206,3.0728853754940713,23.706841432225065,42.83607487237663,
              22.98603351955307,123.8513731825525,14.22217845184164,39.632492113564666,60.776978417266186, 
              14.216184288245717,18.699974509304106,52.70604586518416,12.136664966853646,21.175896700143472, 
              10.224836601307189,1.484744236890332,9.918762362249224,124.25607779578606,4.202840985593106,
              17.50646551724138,52.93091416608514]
    # compute weighted loss
    loss_total = tf.zeros(1)
    for i in range(27):
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true[:,i],
                                                        logits=y_pred[:,i],
                                                        pos_weight=pos_wts[i])
        loss_total += tf.reduce_mean(loss, axis=-1)
    return loss_total#tf.reduce_mean(loss, axis=-1)

def focal_loss(gamma=2., alpha=1.):
    def focal_loss_fixed(y_true, y_pred):
        
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def multilabel_loss(y_true, y_pred):
    
    return K.sum(binary_crossentropy(y_true, y_pred))

class DataGenerator(Sequence):
    """
    Generate data for fit_generator.
    """
    def __init__(self, data_ids, labels, batch_size, n_classes, sample_weights=None, n_channels = 12, dim = 5000, 
                 if_upsamp = True, if_samwts = False, shuffle=True):
        self.data_ids = data_ids
        self.labels = labels #train_y:[n,n_classes+1] where first col is ID
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.dim = dim
        self.shuffle = shuffle
        self.if_samwts = if_samwts
        self.if_upsamp = if_upsamp
        if self.if_samwts:
            self.sample_weights = sample_weights
        #self.pdother = pd.read_csv('otherprocess.csv')
        self.on_epoch_end()



    def __len__(self):
        """
        Denote the number of batches per epoch. 
        """
        return int(len(self.data_ids) / self.batch_size) 
    
    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        data_ids_temp = [self.data_ids[k] for k in indexes]

        # Generate data
        if self.if_samwts:
            X, y, sample_weights = self.__data_generation(data_ids_temp)
            return X, y, sample_weights
        else:
            X, y = self.__data_generation(data_ids_temp)
            return X, y
        
        

        #return X, y, sample_weights 

    def on_epoch_end(self):
        """
        Update indexes after each epoch.
        """
        self.indexes = np.arange(len(self.data_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def __data_generation(self, data_ids_temp):
        'Generates data containing batch_size samples X : (n_samples, *dim, n_channels)' 
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)
        if self.if_samwts:
            sample_weights = np.empty((self.batch_size, ))
        #X2 = np.empty((self.batch_size, 2)) #sex and age
        
        # Generate data
        for i, ID in enumerate(data_ids_temp):
            # Store sample
            if self.if_upsamp:
                X[i,] = np.load('upsampdata2/' + str(ID) + '.npy')
            else:
                X[i,] = np.load('ecgtrain/' + str(ID) + '.npy')
            
            # Store class
            y[i] = self.labels[self.labels[:,0]==ID][0,1:] #first col is ID
            if self.if_samwts:
                sample_weights[i] = self.sample_weights[self.sample_weights[:,0]==ID][0, 1]
            #X2[i] = self.pdother[self.pdother['ID']==ID].iloc[0,1:]
        #Xfi = [X, X2]
        if self.if_samwts:
            return X, y, sample_weights
        else:
            return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)

def model_train(model, train_id, train_label, val_id, val_label, params):
    
       
    # Get class_weight to solve the data imbalanced problem
    counter = {}
    for i in range(1,28):
        counter[i-1] = int(sum(train_label[:,i]))
    
    #class_weight = get_class_weights(train_label.shape[0], counter, params['multiply'])
    #sample_weights = np.zeros((train_label.shape[0],2))
 
    #sample_weights[:,1] = np.log(compute_sample_weight(class_weight, train_label[:,1:])*1e41 )
    #sample_weights[:,0] = train_label[:,0]
    max_val = float(max(counter.values()))       
    class_weight = {class_id : (max_val/num_ecg)**(1) for class_id, num_ecg in counter.items()}

    #[focal_loss(alpha=1, gamma=4)], [focal_loss(alpha=1, gamma=params['multiply'])]
    model.compile(loss=[focal_loss(alpha=1, gamma=params['multiply'])], 
                  optimizer=tfa.optimizers.RectifiedAdam(lr=params["learning_rate"],
                  total_steps=100, warmup_proportion=0.1,min_lr=1e-5), 
                  metrics=['accuracy']) 
    my_callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=3),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, 
                                      min_lr=0.00001, verbose=1),#0.00000001
                    tfa.callbacks.TQDMProgressBar()]#TQDMNotebookCallback(leave_inner=True, leave_outer=True)]
#                     TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)]
    model.fit_generator(generator=DataGenerator(train_id, train_label, 
                                                batch_size=params["batch_size"], 
                                                n_classes=params["disease_num"],
                                                if_upsamp = params["if_upsamp"]),
                                                 #sample_weights = sample_weights),
                         use_multiprocessing=False,
                         workers=45,
                         epochs=100,
                         validation_data=DataGenerator(val_id, val_label, 
                                                       batch_size=params["batch_size"],
                                                       n_classes=params["disease_num"],
                                                       if_upsamp = params["if_upsamp"]),
                                                      # sample_weights = sample_weights),
                         steps_per_epoch=int(len(train_id)/params["batch_size"]),
                         callbacks=my_callbacks, 
                         verbose=1,
                         class_weight=class_weight)
    
    return model

def model_save(model):
    today = date.today()
    # Save the model
    model.save('multilabel_model_' + today.strftime("%m%d") + '.h5')
    # # Save the weights as well
    model.save_weights('multilabel_model_weights_' + today.strftime("%m%d") + '.h5')


def model_load(h5_name):
    # This code can load the whole model
    model = load_model(h5_name)
    # If necesssary, you can create a new model using the weights you have got.
    # Fisrt create a new model...
    # Then load the weights
    # model.load_weights('model_weights_0805.h5')
    return model

def model_eval(model, params):
    def plot_roc(name, labels, predict_prob, cur_clr):
        fp_rate, tp_rate, thresholds = roc_curve(labels, predict_prob)
        roc_auc = auc(fp_rate, tp_rate)
        plt.title('ROC')
        plt.plot(fp_rate, tp_rate, cur_clr, label= name + "'s AUC = %0.4f" % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        
    def plot_confusion_matrix(name, cm, title='Confusion Matrix', cmap='Blues'):
        labels = ['Non-' + name, name]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=30)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
        cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
    
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            plt.text(x_val, y_val, "%0.4f" %(c,), color='black', fontsize=15, va='center', ha='center')
    
    # Visualize the classification result
    # First load the test set into memory
    X_test = []
    y_test = []
    for i in test_id:
        X_test.append(np.load(i))
    for i in range(len(test_id)):
        y_test.append(test_label[test_id[i]])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    test_pos_predict = model.predict(X_test)
    test_predict_onehot = (test_pos_predict >= 0.5).astype(int)

    abbr_list = params["abbr_list"]

    today = date.today()
    # ROC & AUC
    plt.figure(figsize=(24, 20))
    for i in range(len(abbr_list)):
        plt.subplot(5, 6, i+1)
        plot_roc(abbr_list[i], y_test[:, i], test_pos_predict[:, i], 'blue')

    plt.tight_layout()
    plt.savefig('multilabel_roc_' + today.strftime("%m%d") + '.png')
    
    # Confusion matrix
    conf_matrix = []
    for i in range(len(abbr_list)):
        conf_matrix.append(confusion_matrix(y_test[:, i], test_predict_onehot[:, i]))
    plt.figure(figsize=(42, 35))
    for i in range(len(abbr_list)):
        plt.subplot(5, 6, i+1)
        plot_confusion_matrix(abbr_list[i], conf_matrix[i])

    plt.tight_layout()
    plt.savefig('multilabel_conf_' + today.strftime("%m%d") + '.png')
    
def plot_roc(name, labels, predict_prob, cur_clr):
    fp_rate, tp_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(fp_rate, tp_rate)
    plt.title('ROC')
    plt.plot(fp_rate, tp_rate, cur_clr, label= name + "'s AUC = %0.4f" % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    
def plot_confusion_matrix(name, cm, title='', cmap='Blues'):
    labels = ['Non-' + name, name]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=30, fontsize=25)
    plt.yticks(xlocations, labels, rotation=30, fontsize=25)
    plt.ylabel('Committee consensus label', fontsize=25)
    plt.xlabel('Model predicted label', fontsize=25)
    
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        plt.text(x_val, y_val, "%0.3f" %(c,), color='black', fontsize=25, va='center', ha='center')

        
def compute(labels, outputs):
    #y_true: labels, y_pred: outputs
    f1 = f1_score(labels, outputs)

    return f1  

def unit_thresholdf(threshold,labels,outputs):

    predictions = outputs.copy()

    predictions[np.where(predictions >= threshold)] = 1
    predictions[np.where(predictions < threshold)] = 0

    return compute(labels, predictions)

class PostProcessing():


    def __init__(self,fold):

        self.fold = fold

        self.threshold = .5#float(open(f"threshold_{self.fold}.txt", "r").read())#0.5#0.1
        self.metric = compute


    def run(self,predictions):

        predictions_processed = predictions.copy()
        for i in range(27):
            predictions_processed[np.where(predictions_processed[:,i] >= self.threshold[i]), i] = 1
            predictions_processed[np.where(predictions_processed[:,i] < self.threshold[i]), i] = 0
        

        return predictions_processed

    def find_opt_thresold(self, labels, outputs):
        threshold_grid = np.arange(0.05, 0.99, 0.01).tolist()
        threshold_opt = np.zeros((27))
        start = time.time()
        for i in range(27):
            unit_threshold= partial(unit_thresholdf,labels=test_y[:,i],outputs=test_pos_predict[:,i])
            with ProcessPoolExecutor(max_workers=20) as pool:
                result = pool.map(
                     unit_threshold,threshold_grid
            )
            scores = np.array(list(result))
            a = np.where(scores == np.max(scores))
            threshold_opt[i] = threshold_grid[a[0][0]]
        print(f'Processing time: {(time.time() - start)/60}')
        
        return threshold_opt

    def _unit_threshold(self,threshold,labels,outputs):

        predictions = outputs.copy()

        predictions[np.where(predictions >= threshold)] = 1
        predictions[np.where(predictions < threshold)] = 0

        return self.metric(labels, predictions)

    def update_threshold(self,threshold):
        f = open(f"threshold_{self.fold}.txt", "w")
        f.write(str(threshold))
        f.close()
        self.threshold = threshold
        
    def obtain_score(self, labels, outputs):
        f1 = []
        recall = []
        precision = []
        for i in range(27):
            f1.append(compute(test_y[:,i], test_predict_onehot[:,i]) )
            precision.append(precision_score(test_y[:,i], test_predict_onehot[:,i]) )
            recall.append(recall_score(test_y[:,i], test_predict_onehot[:,i]))
        return f1,recall,precision