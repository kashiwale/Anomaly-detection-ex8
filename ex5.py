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

def dataPlot(X,Y,c,figNo = 0,title='',xlabel='',ylabel=''):
    #	label is a string or anything printable with ‘%s’ conversion example 'line1'
    fig = plt.figure(figNo)    
    plt.plot(X,Y,c)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, loc='left')
    return fig
    

def linearRegCostFunction1(theta,X, y,  lambda_):

    m,n = X.shape
    J = 0
    h = X.dot(theta).reshape(m,1)
    b = (h-y)**2
    if(m==0):
        m=0.00001
        J = (1/(2*m))*np.sum(b) + (lambda_/(2*m))*(np.sum(theta[1:]**2))
    else:
        J = (1/(2*m))*np.sum(b) + (lambda_/(2*m))*(np.sum(theta[1:]**2))
        
    non_bias_theta= np.zeros(theta.shape)
    
    non_bias_theta[1:] = theta[1:]
    non_bias_theta= non_bias_theta.reshape((theta.shape[0],1))
    if(m==0):
       m=0.00001 
       grad = (1/m)*(X.T.dot(h-y)) + (lambda_/m)*non_bias_theta
    else:
      grad = (1/m)*(X.T.dot(h-y)) + (lambda_/m)*non_bias_theta  
    
    
    
    # =========================================================================
    
    grad = grad[:]
    return J, grad
    
def linearRegCostFunction(X, y, theta, lambda_):
    m,n = X.shape
    J = 0
    grad = np.zeros(theta.shape)

    h = X.dot(theta).reshape(m,1)
    b = (h-y)**2
    if(m==0):
        m=0.00001
        J = (1/(2*m))*np.sum(b) + (lambda_/(2*m))*(np.sum(theta[1:]**2))
    else:
        J = (1/(2*m))*np.sum(b) + (lambda_/(2*m))*(np.sum(theta[1:]**2))
    
    non_bias_theta= np.zeros(theta.shape)
    
    non_bias_theta[1:] = theta[1:]
    non_bias_theta= non_bias_theta.reshape((theta.shape[0],1))
    if(m==0):
       m=0.00001 
       grad = (1/m)*(X.T.dot(h-y)) + (lambda_/m)*non_bias_theta
    else:
      grad = (1/m)*(X.T.dot(h-y)) + (lambda_/m)*non_bias_theta  
    
    
    
    # =========================================================================
    
    grad = grad[:]
    
    return J, grad
    
def trainLinearReg(X, y, lambda_):
    m,n = X.shape
    initial_theta = np.zeros((n, 1))
    #options = {'full_output': True, 'maxiter': 500}

    #theta, cost, _, _, _ =  op.fmin(lambda t: linearRegCostFunction1(X, y,t,lambda_), initial_theta, **options)
    fmin = minimize(fun=linearRegCostFunction1, x0=initial_theta, args=(X, y, lambda_), method='TNC', jac=True, options={'maxiter': 50})

    #return cost, theta  
    return fmin['fun'],fmin['x'] 
    
def learningCurve(X, y, Xval, yval, lambda_):
    m,n = X.shape
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))
    #cost, theta = trainLinearReg(Xval, yval, lambda_)

    for i in range(m):
        J, theta = trainLinearReg(X[0:i, :], y[0:i], lambda_);
        #J, grad = linearRegCostFunction(X[0:i, :], y[0:i], theta, 0)
        error_train[i] = J
        J, grad = linearRegCostFunction(Xval, yval, theta, 0);
        error_val[i] = J
    
    return error_train, error_val
    
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
    
def plotFit(min_x, max_x, mu, sigma, theta, p, fig = 0):
    #print(min_x,' _ ', max_x)
    x = np.arange((min_x - 15),( max_x + 25), 0.05 ).T
    x= x.reshape(x.shape[0],1)
    #print(x)
    # Map the X values 
    X_poly = polyFeatures(x, p)
    X_poly = np.divide(np.subtract(X_poly,mu),sigma)
    X_poly= np.hstack(((np.ones((X_poly.shape[0],1))), X_poly[:,0:]))
    dataPlot(x, X_poly.dot(theta),'r--',2,title = 'Polynomial Regression Fit (lambda = '+str(lambda_)+')',xlabel='Change in water level (x)',ylabel='Water flowing out of the dam (y)')

def validationCurve(X, y, Xval, yval):
   lambda_vec = np.array([0 ,0.001, 0.003 ,0.01 ,0.03 ,0.1, 0.3, 1, 3 ,10])
   lambda_vec = lambda_vec.reshape((lambda_vec.shape[0],1))
   error_train = np.zeros((lambda_vec.shape[0],1))
   error_val = np.zeros((lambda_vec.shape[0],1))
   for i in range(lambda_vec.shape[0]):
       lambda_ = lambda_vec[i].item()
       dummy,theta = trainLinearReg(X, y, lambda_)
       J, grad = linearRegCostFunction(X, y, theta, 0)
       error_train = np.vstack((error_train,J))
       J, grad = linearRegCostFunction(Xval, yval, theta, 0)
       error_val = np.vstack((error_val,J))
   return lambda_vec, error_train[(lambda_vec.shape[0]):], error_val[(lambda_vec.shape[0]):]
     
# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment  
my_file=r'C:\Users\dkashi200\Yawaris\machine-learning-ex5\ex5\ex5data1.mat'
test = sio.loadmat(my_file)

  
X = test['X'].reshape((test['X'].shape[0],1))
y = test['y']
Xtest = test['Xtest'].reshape((test['Xtest'].shape[0],1))
Xval = test['Xval'].reshape((test['Xval'].shape[0],1))
yval = test['yval']
ytest = test['ytest']

m,n = X.shape 
y=y.reshape(m,1)
fig = dataPlot(X,y,'rx',0,title = 'Test',xlabel='Change in water level (x)',ylabel='Water flowing out of the dam (y)')
fig.show
wait = input("PRESS ENTER TO CONTINUE.")

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
# 

theta = np.array([1 , 1])
theta=theta.reshape((theta.shape[0],1))
x= np.hstack((np.ones((m,1)),X))
J,grad=linearRegCostFunction(x, y, theta, 1)

print('Cost at theta = [1 ; 1]: {}\n(this value should be about 303.993192)\n'.format(J))


## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#


print('Gradient at theta = [1 ; 1]:  [{0},{1}] \n(this value should be about [-15.303016; 598.250744])\n'.format(grad[0].item(), grad[1].item()))

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0

lambda_ = 0
x = np.hstack((np.ones((m,1)),X))
cost, theta = trainLinearReg(x, y, 0)

#  Plot fit over the data

fig = dataPlot(X,y,'rx',0,title = 'Test',xlabel='Change in water level (x)',ylabel='Water flowing out of the dam (y)')

fig = dataPlot(X,np.hstack((np.ones((m,1)),X)).dot(theta),'r--',0)
fig.show

lambda_ = 0
xval = np.hstack((np.ones((Xval.shape[0],1)),Xval))
error_train, error_val = learningCurve(x, y, xval, yval, lambda_)

x_axis= np.arange(m)
fig=dataPlot(x_axis,error_train,'r-',1 )
fig=dataPlot(x_axis,error_val,'b-',1, title = 'Validation',xlabel='Number of Training Samples',ylabel='Error')
#plt.plot(x_axis,error_val,'b-',1)
#plt.show

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly1 = polyFeatures(X, p)
#print(X_poly1[0,:])
X_poly, mu, sigma = featureNormalize(X_poly1)# Normalize
#print(mu[0], sigma[0])
X_poly = np.hstack(((np.ones((X_poly.shape[0],1))), X_poly[:,0:]))                # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = (X_poly_test - mu)/sigma
X_poly_test = np.hstack(((np.ones((X_poly_test.shape[0],1))), X_poly_test[:,0:]))


# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = (X_poly_val - mu)/sigma
X_poly_val= np.hstack(((np.ones((X_poly_val.shape[0],1))), X_poly_val[:,0:]))

print('Normalized Training Example 1:\n');
print(' {}  \n'.format(X_poly[0, :]))

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

lambda_ = 3
cost, theta = trainLinearReg(X_poly, y, lambda_)

theta = theta.reshape((theta.shape[0],1))
#print('theta :',theta)
# Plot training data and fit
dataPlot(X,y,'bx',2,title = 'Polynomial Regression Fit (lambda = '+str(lambda_)+')',xlabel='Change in water level (x)',ylabel='Water flowing out of the dam (y)')
plotFit(min(X), max(X), mu, sigma, theta, p,2)

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#
lambda_vec, error_train, error_val =  validationCurve(X_poly, y, X_poly_val, yval)

p=dataPlot(lambda_vec,error_train,'b-',3,title = 'lambda vs error',xlabel='lambda',ylabel='error')
q=dataPlot(lambda_vec,error_val,'r-',3,title = 'lambda vs error',xlabel='lambda',ylabel='error')

print('lambda\t\tTrain Error\t\t\tValidation Error\n')
for i in range(len(lambda_vec)):
    print(' {}\t\t{}\t\t{}\n'.format(lambda_vec[i].item(), error_train[i].item(), error_val[i].item()))


'''
plt.legend(handles=[p,q],labels=['error','error1'])
p=plt.plot(lambda_vec, error_train)
q=plt.plot( lambda_vec, error_val)
plt.xlabel('lambda')
plt.ylabel('error')
plt.legend([p, q],['error_train', 'error_val'])

figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];  


plot(x-axis, error_train, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
#axis([0 13 0 150])

'''