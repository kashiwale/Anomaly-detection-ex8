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
# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')
ratio = 1.00
figr = 0
# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment  
my_file=r'C:\Users\dkashi200\Yawaris\machine-learning-ex6\ex6\ex6data1.mat'
test = sio.loadmat(my_file)

  
X1 = test['X']
y1 = test['y']
PlotClassData(X1,y1,'bx','ro',figNo = figr)
plt.show()
figr = figr +1

X_train,X_test,y_train,y_test=Train_Test_Sep(X1,y1,1)

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

# Load from ex6data1: 
# You will have X, y in your environment

print('\nTraining Linear SVM ...\n')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_train)
print('Misclassified samples: %d' % (y_train != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_train, y_pred))

plot_decision_regions_reg(X_train, y_train,svc,test_idx=range(40, 51))


## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('\nEvaluating the Gaussian Kernel ...\n')

x1 = np.array([1 ,2 ,1])
x2 = np.array([0 ,4 ,-1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 2 :\n\t{}\n(this value should be about 0.324652)\n'.format(sim))

print('Program paused. Press enter to continue.\n')

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data2: 
# You will have X, y in your environment




my_file=r'C:\Users\dkashi200\Yawaris\machine-learning-ex6\ex6\ex6data2.mat'
test = sio.loadmat(my_file)

  
X1 = test['X']
y1 = test['y']
m,n = X1.shape
y1= y1.reshape(m,)
# Plot training data
PlotClassData(X1,y1,'bx','ro',figNo = figr)
plt.show()
figr = figr +1
## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.
# 
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');


# SVM Parameters

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.
X_train,X_test,y_train,y_test=Train_Test_Sep(X1,y1,0.8)

svc = svm.SVC(C=100, gamma=20, probability=True)

svc.fit(X_train,y_train)  


data_Prob = svc.predict_proba(X_train)[:,0]

y_pred = svc.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


#fig, ax = plt.subplots(figsize=(36,16))  
#ax.scatter(X1[:,0], X1[:,1], s=30, c=data_Prob, cmap='Reds')
X_train,X_test,y_train,y_test=Train_Test_Sep(X1,y1,1)

plot_decision_regions_reg(X_train, y_train,svc,test_idx=range(750, 863))


## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data3: 
# You will have X, y in your environment
my_file=r'C:\Users\dkashi200\Yawaris\machine-learning-ex6\ex6\ex6data3.mat'
test = sio.loadmat(my_file)

  
X1 = test['X']
y1 = test['y']
m,n = X1.shape
y1= y1.reshape(m,)
# Plot training data
PlotClassData(X1,y1,'bx','ro',figNo = figr)
plt.show()
figr = figr +1


X_train,X_test,y_train,y_test=Train_Test_Sep(X1,y1,0.8)
#print('c\t\tg\t\tMisclassified samples\t\tAccuracy')
reg = np.array([0.1,1,3,10,50,100])
sig = np.arange(0.1,10,0.1)
res = np.zeros((reg.shape[0],sig.shape[0]))
colormap = np.array(['r-', 'b-','g-','c-', 'm-', 'y-'])
 
for c in range(reg.shape[0]):
    for g in range(sig.shape[0]):
        svc = svm.SVC(C=reg[c].item(), gamma=sig[g].item(), probability=True)
        svc.fit(X_train,y_train)  
        data_Prob = svc.predict_proba(X_train)[:,0]
        y_pred = svc.predict(X_test)
        res[c,g]   = accuracy_score(y_test, y_pred)
    dataPlot(sig,res[c,:],colormap[c],figr, title = 'Reg :'+str(reg[c]), xlabel = 'Sigma', ylabel = 'accuracy', label = c)
    #figr = figr +1
        #print('{}\t\t{}\t\t{}\t\t{}'.format(c,g,(y_test != y_pred).sum(),accuracy_score(y_test, y_pred)))












