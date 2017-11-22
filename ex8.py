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
from scipy.stats import multivariate_normal as mvnorm
import re
import matplotlib.cm as cm

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

def findClosestCentroids(X, centroids):
    K, m = centroids.shape
    p,q = X.shape
    idx = np.zeros((p, 1))
    for i in range(p):
        test = np.zeros((K,1))
        for j in range(K):
            #print('X :', X[i,:])
            #print('centroids :',centroids[j,:])
            test[j] = norm(X[i,:] - centroids[j,:])**2
        idx[i] = np.argmin(test)
    return idx.astype(int)
    
def computeCentroids(X, idx, K):
    m ,n = X.shape
    centroids = np.zeros((K, n))
    
    for i in np.unique(idx):
        r,s = np.where(idx == i)
        centroids[i,:]=np.mean(X[r],axis = 0)
        
    return centroids

def runkMeans(X, initial_centroids,max_iters  ):
    K, m = initial_centroids.shape
    centroids = initial_centroids
    if(m == 2):
        for i in range(0,max_iters):
            #fig = plt.figure(i) 
            centroids1 = centroids
            idx = findClosestCentroids(X, centroids)
            centroids = computeCentroids(X, idx, K)
            zipped = zip(centroids,centroids1)
            PlotCentroidTrace(zipped,1)
            if(np.array_equal(centroids1 , centroids)):
                break
                #print(centroids)
        PlotDataPoints(X,idx,i,1)
    else:
        for i in range(0,max_iters):
            #fig = plt.figure(i) 
            print(i)
            centroids1 = centroids
            idx = findClosestCentroids(X, centroids)
            centroids = computeCentroids(X, idx, K)
            if(np.array_equal(centroids1 , centroids)):
                break
                #print(centroids)
  
    return idx, centroids

def PlotCentroidTrace(zipped,figNo):
    fig = plt.figure(figNo)    
    d=0
    colormap = np.array(['rx-', 'bs-','g^-','c+-', 'mo-', 'yo-','rx-', 'bs-','g^-','c+-', 'mo-', 'yo-','rx-', 'bs-','g^-','c+-', 'mo-', 'yo-'])
    #Xcolormap = np.array(['gx', 'ys','m^','c+', 'md', 'yo','gx', 'ys','m^','c+', 'md', 'yo','gx', 'ys','m^','c+', 'md', 'yo','gx', 'ys','m^','c+', 'md', 'yo'])
    for p , q in zipped:
        #print(p , ' - ', q, ' -- ',d)
        #if(d == 0):
            #print(p , ' - ', q, ' -- ',d)
        plt.plot((p[0],q[0]),(p[1],q[1]),colormap[d])
        d+=1
        
def PlotDataPoints(X,idx,iter_num, figNo):
    fig = plt.figure(figNo)    
    Xcolormap = np.array(['gx', 'ys','m^','c+', 'md', 'yo','gx', 'ys','m^','c+', 'md', 'yo','gx', 'ys','m^','c+', 'md', 'yo','gx', 'ys','m^','c+', 'md', 'yo'])
    for w in np.unique(idx):
        plt.plot(X[np.where(idx == w)[0]][:,0],X[np.where(idx == w)[0]][:,1],Xcolormap[w])
        plt.title('iteration # :' + str(iter_num), loc='left')

def pca(x):
    m , n = x.shape
    U = np.zeros(n)
    S = np.zeros(n)
    Xcov = x.T.dot(x)/m
    U,S,temp = np.linalg.svd(Xcov)
    return U, S

def projectData(X, U, K):
    #PROJECTDATA Computes the reduced data representation when projecting only 
    #on to the top k eigenvectors
    #   Z = projectData(X, U, K) computes the projection of 
    #   the normalized inputs X into the reduced dimensional space spanned by
    #   the first K columns of U. It returns the projected examples in Z.
    #
    
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the projection of the data using only the top K 
    #               eigenvectors in U (first K columns). 
    #               For the i-th example X(i,:), the projection on to the k-th 
    #               eigenvector is given as follows:
    #                    x = X(i, :)';
    #                    projection_k = x' * U(:, k);
    #
    Z = X.dot(U[:,:K])
    return Z

def recoverData(Z, U, K):
    #RECOVERDATA Recovers an approximation of the original data when using the 
    #projected data
    #   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
    #   original data that has been reduced to K dimensions. It returns the
    #   approximate reconstruction in X_rec.
    #
    
    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the approximation of the data by projecting back
    #               onto the original space using the top K eigenvectors in U.
    #
    #               For the i-th example Z(i,:), the (approximate)
    #               recovered data for dimension j is given as follows:
    #                    v = Z(i, :)';
    #                    recovered_j = v' * U(j, 1:K)';
    #
    #               Notice that U(j, 1:K) is a row vector.
    #               
    X_rec = Z.dot(U[:,:K].T)
    return X_rec
def image_array_disp(X):
    m,n = X.shape
    img_size = math.ceil(m**(0.5))
    f, axarr = plt.subplots(img_size, img_size,figsize=(10,10))
    for i in range(img_size):
        for j in range(img_size):
            k=img_size*i + j
            #a=fig.add_subplot(i+1,j+1,1)
            axarr[i, j].imshow(X[k].reshape(32,32).T)
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
        
    plt.show()

def plotDataPoints(X, idx, K):
    #PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    #index assignments in idx have the same color
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
    #   with the same index assignments in idx have the same color
    
    # Create palette
    palatte1 = np.zeros((K,4))
    for i in range(K):
        palatte1[i,:] = cm.hsv(i/K)
    #print(palatte1.shape)
    #print(idx.shape)
    color1 = palatte1[idx].reshape(1000,4)
    #print(color1.shape)
    fig = plt.figure(4)    
    plt.scatter(X[:,0], X[:,1], 15, color1)
    
def estimateGaussian(X):
    m,n = X.shape
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))    
    mu = np.mean(X,axis = 0)
    sigma2 = np.std(X,axis =0)**2
    return mu, sigma2
    
def multivariateGaussian(X, mu, Sigma2):
    #k = len(mu)
    m = len(Sigma2)
    n = Sigma2.size/m
    #Sigma2 = Sigma2.reshape(m,n)
    if(m==1 or n ==1):
        Sigma2 = np.diag(Sigma2)
    #c =  np.linalg.pinv(Sigma2)
    #print(c.shape)
    #b=X-mu
    #print(b.shape)
    #d=np.exp(np.sum((b.dot(c)*b),axis = 1)*(-1)/2)/((((2*np.pi)**k )*np.linalg.det(c))**(0.5))
    d = mvnorm.pdf(X,mu,sigma2)    
    return d
    
def visualizeFit(X, mu, sigma2,resolution=0.02):
        # setup marker generator and color map
    #markers = ('s', 'x', 'o', '^', 'v')
    levels = [ -0.001,-0.0001,-0.00001,-0.0000001, 0,0.0000001,0.00001,0.0001,0.001]
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan','orange','magenta')
    cmap = ListedColormap(colors[:])
    fig, ax = plt.subplots(figsize=(10,8))  
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    print(xx1.shape)
    print(xx2.shape)
    #print(Z)
    Z = multivariateGaussian((np.vstack((xx1.ravel(), xx2.ravel())).T).reshape(xx1.shape[0]*xx1.shape[1],2),mu,sigma2)
    print(Z)    
    plt.contourf(xx1, xx2, Z.reshape(xx2.shape),levels, alpha=0.4, cmap=cmap)
    plt.plot(X[:,0],X[:,1],'bo')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
                           
def selectThreshold(yval1, pval1):
    bestEpsilon = 0
    bestF1 = 0
    F1_ = 0
    stepsize = (np.max(pval1) - np.min(pval1)) / 1000
    for epsilon in np.arange(min(pval1),max(pval1),stepsize):
        
        test = np.hstack((yval1.reshape(yval1.size,1),pval1.reshape(pval1.size,1)))
        predictions = test[test[:,1]<epsilon][:,0]
        tp = np.sum(predictions)
        fp = predictions.shape[0] - tp
        fn = np.sum(test[test[:,1]>=epsilon][:,0])
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1_=2*prec*rec/(prec + rec)
        if F1_ > bestF1:
            bestF1 = F1_
            bestEpsilon = epsilon
    return bestEpsilon, bestF1
    
    
cmaps = [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3', 'Vega10',
                             'Vega20', 'Vega20b', 'Vega20c']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]


## Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easy to
#  visualize.
#
#  Our example case consists of 2 network server statistics across
#  several machines: the latency and throughput of each machine.
#  This exercise will help us find possibly faulty (or very fast) machines.
#

print('Visualizing example dataset for outlier detection.\n\n');

#  The following command loads the dataset. You should now have the
#  variables X, Xval, yval in your environment

fig = plt.figure(0)
my_file = r'C:\Users\dkashi200\Yawaris\machine-learning-ex8\ex8\ex8data1.mat'
test = sio.loadmat(my_file)
X1 = test['X']
yval=test['yval']
Xval = test['Xval']
plt.plot(X1[:,0],X1[:,1],'bo')
plt.xlim([0,30])
plt.ylim([0,30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')

## ================== Part 2: Estimate the dataset statistics ===================
#  For this exercise, we assume a Gaussian distribution for the dataset.
#
#  We first estimate the parameters of our assumed Gaussian distribution, 
#  then compute the probabilities for each of the points and then visualize 
#  both the overall distribution and where each of the points falls in 
#  terms of that distribution.
#
print('Visualizing Gaussian fit.\n\n')

#  Estimate my and sigma2
mu, sigma2 = estimateGaussian(X1)

#  Returns the density of the multivariate normal at each data point (row) 
#  of X
p = multivariateGaussian(X1, mu, sigma2)

visualizeFit(X1,  mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')

## ================== Part 3: Find Outliers ===================
#  Now you will find a good epsilon threshold using a cross-validation set
#  probabilities given the estimated Gaussian distribution
#

pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: {0:e}\n'.format(epsilon))
print('Best F1 on Cross Validation Set:  {0:f}\n'.format(F1))
print('   (you should see a value epsilon of about 8.99e-05)\n\n');

#  Find the outliers in the training set and plot the
#outliers = find(p < epsilon);

#  Find the outliers in the training set and plot the
outliers = np.where(p < epsilon)

#  Draw a red circle around those outliers

plt.plot(X1[outliers][:,0], X1[outliers][:,1], 'ro', linewidth= 2, markersize= 10)


## ================== Part 4: Multidimensional Outliers ===================
#  We will now use the code from the previous part and apply it to a 
#  harder problem in which more features describe each datapoint and only 
#  some features indicate whether a point is an outlier.
#

#  Loads the second dataset. You should now have the
#  variables X, Xval, yval in your environment
fig = plt.figure(1)
my_file = r'C:\Users\dkashi200\Yawaris\machine-learning-ex8\ex8\ex8data2.mat'
test = sio.loadmat(my_file)
X2 = test['X']
yval=test['yval']
Xval = test['Xval']

#  Apply the same steps to the larger dataset
mu, sigma2 = estimateGaussian(X2)

#  Training set 
p = multivariateGaussian(X2, mu, sigma2)

#  Cross-validation set
pval1 = multivariateGaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = selectThreshold(yval, pval1)

print('Best epsilon found using cross-validation: {0:e}\n'.format(epsilon))
print('Best F1 on Cross Validation Set:  {0:f}\n'.format(F1))
print('   (you should see a value epsilon of about 1.38e-18)\n\n');
'''
print('Best epsilon found using cross-validation: #e\n', epsilon);
print('Best F1 on Cross Validation Set:  #f\n', F1);
print('# Outliers found: #d\n', sum(p < epsilon));
print('   (you should see a value epsilon of about 1.38e-18)\n\n');
'''
