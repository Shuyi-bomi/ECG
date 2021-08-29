# ECG
Multilabel ECG Classification

## Background:

Cardiovascular diseases are the leading cause of death globally, evidence proves that early diagnosis of different cardiac abnormalities can help clinicians provide timely interventions and provide better treatment effect. Deep-learning based methods gains more popularity when diagnosing multiple cardiac abnormalities using 12-lead ECG data due to its efficiency and flexibility. Here we also try to develop deep neural network for multi-label classification.


## Data:
The dimension of our data will be #of samples×5000×12, from which 5000 is composed of 500HZ times 10 seconds recording. 

## Experiment:
(Refer to code ‘modelbuild.py’ and ‘model_training.py’)

### Deep Neural Network

#### Neural Network Architecture:
We designed an improved ResNet to assign the 12-lead ECG recordings into the 27 diagnostic classes. The improved ResNet consists of one convolutional layer followed by N = 8 residual blocks (ResBs), each of which contain two convolutional layers and a squeeze and excitation (SE) block. After 8 residual blocks we apply 2 dense layers with ‘sigmoid’ activation function for the final dense layer of 27 nodes. The reason we choose ‘sigmoid’ instead of  ‘softmax’  activation function is because is due to the fact that some samples have multiple classes of 27 clinical diagnoses. So we assumed that each class was independent and used the sigmoid function for each output neurons to cope with this multi-label classification problem.

#### Loss function:
(Refer to code ‘modelbuild.py’ )
1: Binary cross entropy for each class and take the mean, in order to deal with data imbalance problem, when we train the model using Keras, we set pre-specified different weight for 27 classes:
Pre-specified weight_i = # of samples which belongs to class j/# of samples which belongs to class I
Class j is the class that has most samples
2: Focal loss(parameter alpha and beta which to be tuned)
3: Weighted Binary cross entropy(assign weight by ourselves)

#### Optimizer:
We use ‘RectifiedAdam’ optimizer with learning rate warm up and cool down technique.

### Data resampling:
We adopt data resampling to deal with data imbalance issue. In detail, we aim to obtain a more balanced data distribution by oversampling. We didn’t try undersampling in case losing information invaluable to a model. We use random oversampling to increase examples from the minority class in the training dataset by adding noise of the existing samples. 
(Refer to code ‘multi-label.ipynb’)

### Post-processing (Threshold optimization):
(Refer to code ‘postprocessing.py’,’threshold.py’)
After we obtain the model and predict on the test dataset, we will obtain probability from 0 to 1. Afterwards, we didn’t adopt the usual threshold 0.5 to decide if the sample belongs to that class or not. Instead, we learn the threshold for each class from the validation dataset and apply to the test dataset. This is proved to be useful when dealing with imbalance issue and for neural network model. 

### Ensemble Learning:
(Refer to code ‘model_training.py’ and previous procedure splitting data into 5 groups, each includes one training and one validation dataset) 
To improve the robustness of the classification task, we created an ensemble of five neural network models trained via five-fold cross validation. The threshold of each model was optimized by its split validation set, and ECGs were classified according to the majority vote.



## Result:

1: From the result on training dataset, we could see after threshold optimization the validation dataset actually obtained the excellent performance: We calculate binary f1 score for each class and obtained f1 scores >=0.79 for all classes. (Refer to ‘score.json’ to check details). For the test dataset, we obtain weighted f1 = 0.64. (Refer to ‘scoretest.json’ to check details). So I believe threshold optimization works and we need to choose proper criterion to obtain threshold in the future.

2: From the roc curve and auc value, I do believe our model learns from the dataset. Whereas the data imbalance problem is extremely serious for many minority classes and the lowest positive rate could be 0.0073. So model is hard to assign high probability for some classes, which makes false negative number high. 
<p align="middle">
  <img src="https://github.com/Shuyi-bomi/ECG/blob/master/result/multilabel_roc_072114.png" width="600" />
</p>

3: I tried random oversampling by adding noise of the existing samples for the minority class but this didn’t work from the result. But I still believe this is necessary in this scenario and we should try more rigorous data resampling method. 








