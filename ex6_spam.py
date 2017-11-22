# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:25:12 2016

@author: dkashi200
"""
from IPython.display import Image
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm 

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize
from numpy import newaxis, r_, c_, mat, e
import scipy.optimize as op
import scipy.io as sio
import matplotlib.image as mpimg
import math
from scipy.linalg import norm
import re

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import pandas as pd

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions_reg(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    fig, ax = plt.subplots(figsize=(10,8))  
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    #xx1_std=Standardise(xx1)
    #xx2_std=Standardise(xx2)
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
                    
def dataPlot(X,Y,c,figNo = 0,title='',label='',xlabel='',ylabel=''):
    #	label is a string or anything printable with ‘%s’ conversion example 'line1'
    fig = plt.figure(figNo)    
    plt.plot(X,Y,c,label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, loc='left')
    return fig
    


    
def polyFeatures(X, p):
    m,n = X.shape
    X_poly = np.array([])
    
    for i in range(p):
        j=i+1
        if(i == 0):
            X_poly= np.power(X,j)
        else:
            X_poly = np.hstack((X_poly,np.power(X,j)))
        j=0
    #print('value of X: {}'.format(X)) 
    #print('value of X**2 : {}'.format(X_poly[:,0]))
    return X_poly

       
def featureNormalize(X):
    # You need to set these values correctly
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1 , X.shape[1]))
    mu = np.mean(X,axis=0)
    
    X_norm = np.subtract(X , mu)
    sigma = np.std(X_norm, axis = 0)
    X_norm = np.divide(X_norm , sigma)
    return X_norm, mu, sigma
    

def PlotClassData(X,Y,cpos,cneg,figNo = 0,label1='', label2='',xlabel='',ylabel=''):
    fig = plt.figure(figNo)
    pos=np.nonzero(Y==1)[0].reshape((len(np.nonzero(Y==1)[0]),1))[:,0]
    neg=np.nonzero(Y==0)[0].reshape((len(np.nonzero(Y==0)[0]),1))[:,0]
    dataPlot(X[pos,0],X[pos,1],cpos,label=label1,xlabel=xlabel,ylabel=ylabel)
    dataPlot(X[neg,0],X[neg,1],cneg,label=label2,xlabel=xlabel,ylabel=ylabel)
    plt.legend(loc='upper center', shadow=True, fontsize='x-large')  
    
    
def gaussianKernel(x1, x2, sigma):
    sim = e**((-1/(2*(sigma**2)))*(norm(x1 - x2)**2))
    return sim

def Train_Test_Sep(X,y,ratio):
    m,n = X.shape
    m_train = int(m*ratio)
    m_test = m - int(m*ratio)
    rand_indices = np.arange(X.shape[0])
    np.random.shuffle(rand_indices)
    X_train=X1[(rand_indices[0:m_train]),:]
    y_train=y1[(rand_indices[0:m_train])].reshape(m_train,)
    X_test=X1[(rand_indices[m_train:]),:]
    y_test=y1[(rand_indices[m_train:])].reshape(m_test,)
    
    return X_train,X_test,y_train,y_test
    
def PreProcess(s):
    s = s.lower()
    s = re.sub('>|/>|<','', s)
    s = re.sub('[0-9]+','number', s)
    s = re.sub('[$]+',' dollar ', s)
    s = re.sub('(http|https)://.*(com|/)','httpaddr', s)
    s = re.sub('([a-z-A-Z0-9]*)(@)[a-z-A-Z0-9.]+','emailaddr', s)
    s = s.replace('\n','')
    b = re.split('[@:,_&:#%`~\s/\?\(\)\[\]\+\-\.\{\}\'\"\=\$\^\*\!\n]',s)
    c = list(filter(lambda x: len(x) > 0,b))
    return c
    
def EmailProcess(EmailFile,VocabFile):
    with open(EmailFile, 'r') as f:
    #f = open("months.txt")
        a = f.read()
        c= PreProcess(a)
        #print(c)
    ps = PorterStemmer()
    c=pd.DataFrame(c)
    c['stem']=c[:][0].apply(lambda x: ps.stem(x))
    print_Stemmed_Email(c['stem'])
    data = pd.read_csv(VocabFile, sep="\t", header = None) 
    n = len(data)
    f=pd.merge(data, c, how='inner', on=None, left_on=[1], right_on=['stem'],
           left_index=False, right_index=False, sort=True,
           suffixes=('_x', '_y'), copy=True, indicator=False) 
    f=f.rename( columns={'0_x': 'word_index'})
    
    return f['word_index'],n, data,c
    
def emailFeatures(word_indices, n):
    x = np.zeros((1, n))
    for i in word_indices:
        x[:,i] = 1
    return x
    
def print_Stemmed_Email(x):
    l = 0
    k= ''
    for i in range(len(x)):
        k = k + str(x[i]) + ' '
        l = l + len(k)
        if (l > 700) or (i == len(x)-1):
            print(k)
            l = 0
            k = ''
        
    
## Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

print('\nPreprocessing sample email (emailSample1.txt)\n')


ratio = 1.00
figr = 0
# Extract Features
my_file1=r'C:\Users\dkashi200\Yawaris\machine-learning-ex6\ex6\emailSample3.txt'
my_file2=r'C:\Users\dkashi200\Yawaris\machine-learning-ex6\ex6\vocab.txt'
word_index , n, vocab, C= EmailProcess(my_file1,my_file2)
print('word_index :\n')
print_Stemmed_Email(word_index)

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n. 
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

print('\nExtracting features from sample email (emailSample1.txt)\n')

features     = emailFeatures(word_index,n)

# Print Stats
print('Length of feature vector: {}\n'.format(len(features)))
print('Number of non-zero entries: {}\n'.format(np.sum(features > 0)))

## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
my_file = r'C:\Users\dkashi200\Yawaris\machine-learning-ex6\ex6\spamTrain.mat'
test = sio.loadmat(my_file)

X1 = test['X']
y1 = test['y']

X_train,X_test,y_train,y_test=Train_Test_Sep(X1,y1,ratio)
print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

C = 0.1
svc = svm.LinearSVC(C=0.1, loss='hinge', max_iter=1000)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_train)
print_Stemmed_Email(y_pred)
print('Misclassified samples: %d' % (y_train != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_train, y_pred))



## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
my_file=r'C:\Users\dkashi200\Yawaris\machine-learning-ex6\ex6\spamTest.mat'
test = sio.loadmat(my_file)

X1 = test['Xtest']
y1 = test['ytest']

X_train,X_test,y_train,y_test=Train_Test_Sep(X1,y1,ratio)
print('\nEvaluating the trained Linear SVM on a test set\n')
print('(this may take 1 to 2 minutes) ...\n')

y_pred = svc.predict(X_train)

print('Misclassified samples for test: %d' % (y_train != y_pred).sum())

print('Accuracy for test: %.2f' % accuracy_score(y_train, y_pred))

## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtin the vocabulary list

weights = pd.DataFrame(svc.coef_.T)
Weight_Dist=pd.merge(vocab, weights, how='inner', on=None, left_on=None, right_on=None,
           left_index=True, right_index=True, sort=True,
           suffixes=('_x', '_y'), copy=True, indicator=False).rename( columns={'1_x': 'word','0_y': 0,'1_y': 1 })
           
Weight_Dist=Weight_Dist.sort_values(0,ascending=False)[:20][[1,0]]
print('Top predictors of spam:\n {} '.format(Weight_Dist))

## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
#  The following code reads in one of these emails and then uses your 
#  learned SVM classifier to determine whether the email is Spam or 
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!

my_file1=r'C:\Users\dkashi200\Yawaris\machine-learning-ex6\ex6\spamSample1.txt'
word_index , n, vocab, C= EmailProcess(my_file1,my_file2)
#print('word_index :\n{}'.format(word_index))

print('\nExtracting features from sample email (spamSample1.txt)\n')

features     = emailFeatures(word_index,n)

# Print Stats
print('Length of feature vector: {}\n'.format(len(features)))
print('Number of non-zero entries: {}\n'.format(np.sum(features > 0)))

y_pred = svc.predict(features)

print(y_pred.item())
