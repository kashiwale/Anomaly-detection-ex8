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

def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions_reg(X, y, classifier, test_idx=None, resolution=0.02, degree = 6):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    #xx1_std=Standardise(xx1)
    #xx2_std=Standardise(xx2)
    Z = classifier.predict(mapFeature(xx1.ravel(),xx2.ravel(), degree)[:,1:])
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
                    
#def sigmoid(z):
#    return 1.0 / (1.0 + np.exp(-z))
    
def pause():
    input("Press the <ENTER> key to continue...")

def dataPlot(X,Y,c,label='',xlabel='',ylabel=''):
    #	label is a string or anything printable with ‘%s’ conversion example 'line1'
    plt.plot(X,Y,c,label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def PlotClassData(X,Y,cpos,cneg):
    pos=np.nonzero(Y==1)[0].reshape((len(np.nonzero(Y==1)[0]),1))[:,0]
    neg=np.nonzero(Y==0)[0].reshape((len(np.nonzero(Y==0)[0]),1))[:,0]
    dataPlot(X[pos,0],X[pos,1],cpos,label='Admitted',xlabel='Exam 1 score',ylabel='Exam 2 score')
    dataPlot(X[neg,0],X[neg,1],cneg,label='Not admitted',xlabel='Exam 1 score',ylabel='Exam 2 score')
    plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    # Put a nicer background color on the legend.
    #legend.get_frame().set_facecolor('#00FFCC')
    #plt.show()

def featureNormalize(X):
    # You need to set these values correctly
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1 , X.shape[1]))
    mu = np.mean(X,axis=0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma
    
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - (alpha/m)*(X.T.dot((X.dot(theta) - y)))
        # Save the cost J in every iteration
        J_history[iter] = computeCostMulti(X, y, theta)
    return theta, J_history
def computeCostMulti(X, y, theta):
    m = len(y)
    J = 0
    J = (1/(2*m)) * np.sum((X.dot(theta) - y)**2)
    return J

def normalEqn(X, y):
    theta = theta = np.zeros((X.shape[1], 1))
    #theta = pinv(X'*X)*X'*y
    theta = la.pinv(X.T.dot(X)).dot(X.T.dot(y))
    return theta
    
def sigmoid(z):
    g = np.zeros((z.shape))
    g = 1/(np.exp(-z) +1 )
    return g

def costFunction1(theta, X, y):
    #Initialize some useful values
    m,n = X.shape
    
    J = 0
    
# logistic hypothesis z    
    z = X.dot(theta).reshape(X.shape[0],1)
#hyothesis
    h = sigmoid(z)
    true_val = -y * np.log(h)
    false_val = -(1 - y) * np.log(1 -h)
    J =  np.sum(true_val +  false_val)/m
    
#gradient    
    #grad = np.sum((h - y) * X,axis = 0)/m
    
    
    return J##, grad
    
def plotDecisionBoundary(theta, X, y):
    PlotClassData(X[:,1:3],y,'bx','ro')
    
    if(X.shape[1] <= 3):
        plot_X = r_[X[:,2].min() - 2, X[:,2].max() + 2]
        plot_y = (-1./theta[2]) * (theta[1]*plot_X + theta[0])
        plt.plot(plot_X, plot_y)
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
    else:
        pass
'''    
    #% Here is the grid range
        u = np.linspace(-1, 1.5, 50);
        v = np.linspace(-1, 1.5, 50);
    
        z = np.zeros((u.shape[0], v.shape[0]));
        #% Evaluate z = theta*x over the grid
        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                z(i,j) = mapFeature(u[i], v[j])*theta;
            
        
        z = z'; % important to transpose z before calling contour
    
        #% Plot z = 0
        #% Notice you need to specify the range [0, 0]
        contour(u, v, z, [0, 0], 'LineWidth', 2)
'''
    
def costFunction(theta, X, y):
    #Initialize some useful values
    m,n = X.shape
    
    J = 0
    grad = np.zeros((theta.shape))
# logistic hypothesis z    
    z = X.dot(theta)
#hyothesis
    h = sigmoid(z)
    true_val = -y * np.log(h)
    false_val = -(1 - y) * np.log(1 -h)
    J =  np.sum(true_val +  false_val)/m
#gradient    
    grad = np.sum((h - y) * X,axis = 0)/m
    
    
    return J, grad
    
def mapFeature(X1, X2, degree = 6):
    #degree = 6
    #out = np.ones((X1.shape[0],1));
    out=np.hstack((np.ones((len(X1),1))    ,X1.reshape((len(X1),1))))
    for i in range(degree):
        for j in range(i):
            out = np.c_[out,(X1**(i-j))*(X2**j)];
    return out

    
    
def CostFunc(theta,x,y):
    m,n = x.shape; 
    theta = theta.reshape((n,1));
    y = y.reshape((m,1));
    term1 = np.log(sigmoid(x.dot(theta)));
    term2 = np.log(1-sigmoid(x.dot(theta)));
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2;
    J = -((np.sum(term))/m);
    return J;
    

def Standardise(p):
    sc = StandardScaler()
    sc.fit(p)
    return sc.transform(p)
    
def predict(Theta1, Theta2, X):
    m = X.shape[0]
    #num_labels = Theta2.shape[0]
    p = np.zeros((m, 1))

    h1 = sigmoid(np.hstack((np.ones((m,1)),X)).dot(Theta1.T))
    h2 = sigmoid(np.hstack((np.ones((m,1)),h1)).dot(Theta2.T))
    p = h2.max(1)
    return p    

def costFunctionReg1(theta, X, y, lambda_):
    #Initialize some useful values
    #print('y',y.shape)
    m,n = X.shape
    theta=theta.reshape(n,1)
    
    J = 0

# logistic hypothesis z    
    z = X.dot(theta)
#hyothesis
    h = sigmoid(z).reshape(m,1)
    true_val = -y * np.log(h)
    
    false_val = -(1 - y) * np.log(1 -h)
    
    #theta_reg = (lambda/(2*m))*sum(([0;theta(2:size(theta))].^2));
    theta_reg = (lambda_/(2*m))*(np.sum(theta[1:]**2))
    
    J =  np.sum(true_val +  false_val)/m + theta_reg
    
    return J

def Gradient(theta, X, y, lambda_):
    #Initialize some useful values
    m,n = X.shape
    
    theta = theta.reshape(n,1)
    grad = np.zeros((theta.shape))
# logistic hypothesis z    
    z = X.dot(theta)
#hyothesis
    h = sigmoid(z).reshape(m,1)

    #theta_reg = (lambda/(2*m))*sum(([0;theta(2:size(theta))].^2));

    #J =  np.sum(true_val +  false_val)/m + theta_reg
#gradient   
    #p= (np.sum((h - y) * X,axis = 1)/m).reshape((1,X.shape[1]))
    p=np.zeros((m,1))
    q=np.zeros((m,1))
    p= (np.sum(((h - y) * X)/m,axis = 0))
    q = (lambda_/(m)) * (np.c_[0,theta[1:].reshape(1,n-1)])
    grad = np.add(p , q)
    return  grad

def image_array_disp(X):
    m,n = X.shape
    img_size = math.ceil(m**(0.5))
    f, axarr = plt.subplots(img_size, img_size)
    for i in range(img_size):
        for j in range(img_size):
            k=img_size*i + j
            #a=fig.add_subplot(i+1,j+1,1)
            axarr[i, j].imshow(X[k].reshape(20,20).T)
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
        
    plt.show()
    
def oneVsAll(X, y, num_labels, lambda_):
    m,n = X.shape
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))
    # Add ones to the X data matrix
    X = np.hstack((np.ones((X.shape[0],1)),X))
    # Set Initial theta
    initial_theta = np.zeros((n + 1, 1))
    for c in range(1,(num_labels+1)):
        result = op.minimize(fun = costFunctionReg1,  x0 = initial_theta, args = ( X, (y==c).astype(int),lambda_),method = 'TNC', jac = Gradient)
        print ('Opt Cost for %f: %f' % (c,result.fun))
        all_theta[c-1,:] = result.x
    return  all_theta

def predictOneVsAll(all_theta, X):
    m,n = X.shape
    #num_labels = all_theta.shape[0]
    r=np.hstack((np.ones((X.shape[0],1)),X)).dot(all_theta.T)
    p=np.argmax(r,axis=1) +1 
    return p.reshape(p.shape[0],1)
    
def predict_nn(Theta1, Theta2, X):
    m,n = X.shape
    p = np.zeros((m,1))
    X = np.hstack((np.ones((m,1)),X))
    z2=X.dot(Theta1.T)
    a2 = sigmoid(z2)
    
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    p=np.argmax(a3,axis=1)+1
    return p.reshape(m,1)


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lambda_):
    Theta1 = nn_params[0:( hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1),order='F')
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1,order='F')
    #print('nn_Cost Theta1 : {}'.format(Theta1))  
    #print('nn_Cost Theta2 : {}'.format(Theta2)) 
    m,n = X.shape
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    a1 = np.hstack((np.ones((m,1)),X))
    z2=a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    #print(a3.shape)
    

    r=np.zeros((num_labels,m))
    
    for i in range(m):
        r[y[i]-1,i]=1
    #print(r.shape)
    w_log = np.zeros(a3.shape)
    w_log_I = np.zeros(a3.shape)
    
    w_log = np.log(a3);
    w_log_I = np.log(1-a3);
    
    for i in range(m):
        J = J+np.sum(-r[:,i].dot(w_log[i,:]) - (1-r)[:,i].dot(w_log_I[i,:]))/m
    #print(J)
    reg = lambda_*(np.sum(Theta1[:,1:]**2)+np.sum(Theta2[:,1:]**2))/(2*m)
    J = J + reg
    #print(J)
    
    #####BP
    for t in range(m):
#step1
        a_1 = np.r_[1,X[t,:]].reshape((1,n+1))
        z_2= a_1.dot(Theta1.T)
        a_2 = sigmoid(z_2)
        #print('a_2 shape : {0}'.format(a_2.shape))
        a_2 = np.c_[1,a_2]
        z_3 = a_2.dot(Theta2.T)
        a_3 = sigmoid(z_3)        
        #print('a_3 shape : {0}'.format(a_3.shape))
#step2  
        delta3 = a_3 - r[:,t]
        z_2 = np.c_[1,z_2] 
        #print('z_2 shape : {0}'.format(z_2.shape))
        #print('delta3 shape : {0}'.format(delta3.shape))
        #print('Theta2 shape : {0}'.format(z_2.shape))
        delta2=((delta3).dot(Theta2))*sigmoidGradient(z_2)
        #print('delta2 shape : {0}'.format(delta2.shape))
        delta2 = delta2[:,1:]
        #print('delta2 shape : {0}'.format(delta2.shape))
        #print('a_1 shape : {0}'.format(a_1.shape))
        #print('Theta1_grad shape : {0}'.format(Theta1_grad.shape))
        #print('a_2 shape : {0}'.format(a_2.shape))
        #print('Theta2_grad shape : {0}'.format(Theta2_grad.shape))
        Theta1_grad = Theta1_grad + (delta2.T).dot(a_1)
        Theta2_grad = Theta2_grad + (delta3.T).dot(a_2)

    
    
    Theta1_grad = Theta1_grad/m
    Theta2_grad = Theta2_grad/m
    
#Regularised
    reg_Theta1 = Theta1[:,1:]*(lambda_/m)
    reg_Theta2 = Theta2[:,1:]*(lambda_/m)
    
    
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + reg_Theta1 
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + reg_Theta2 
    
    
    
    

    
# Unroll gradients
#    grad = [Theta1_grad(:) ; Theta2_grad(:)];
    grad = np.hstack([Theta1_grad.T.ravel(), Theta2_grad.T.ravel()])
    


    return J, grad
    
def nnCostFunction1(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lambda_):
    Theta1 = nn_params[0:( hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)
    #print('nn_Cost Theta1 : {}'.format(Theta1))  
    #print('nn_Cost Theta2 : {}'.format(Theta2)) 
    m,n = X.shape
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    a1 = np.hstack((np.ones((m,1)),X))
    z2=a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    #print(a3.shape)
    

    r=np.zeros((num_labels,m))
    
    for i in range(m):
        r[y[i]-1,i]=1
    #print(r.shape)
    w_log = np.zeros(a3.shape)
    w_log_I = np.zeros(a3.shape)
    
    w_log = np.log(a3);
    w_log_I = np.log(1-a3);
    
    for i in range(m):
        J = J+np.sum(-r[:,i].dot(w_log[i,:]) - (1-r)[:,i].dot(w_log_I[i,:]))/m
    #print(J)
    reg = lambda_*(np.sum(Theta1[:,1:]**2)+np.sum(Theta2[:,1:]**2))/(2*m)
    J = J + reg
    #print(J)
    
    #####BP
    for t in range(m):
#step1
        a_1 = np.r_[1,X[t,:]].reshape((1,n+1))
        z_2= a_1.dot(Theta1.T)
        a_2 = sigmoid(z_2)
        #print('a_2 shape : {0}'.format(a_2.shape))
        a_2 = np.c_[1,a_2]
        z_3 = a_2.dot(Theta2.T)
        a_3 = sigmoid(z_3)        
        #print('a_3 shape : {0}'.format(a_3.shape))
#step2  
        delta3 = a_3 - r[:,t]
        z_2 = np.c_[1,z_2] 
        #print('z_2 shape : {0}'.format(z_2.shape))
        #print('delta3 shape : {0}'.format(delta3.shape))
        #print('Theta2 shape : {0}'.format(z_2.shape))
        delta2=((delta3).dot(Theta2))*sigmoidGradient(z_2)
        #print('delta2 shape : {0}'.format(delta2.shape))
        delta2 = delta2[:,1:]
        #print('delta2 shape : {0}'.format(delta2.shape))
        #print('a_1 shape : {0}'.format(a_1.shape))
        #print('Theta1_grad shape : {0}'.format(Theta1_grad.shape))
        #print('a_2 shape : {0}'.format(a_2.shape))
        #print('Theta2_grad shape : {0}'.format(Theta2_grad.shape))
        Theta1_grad = Theta1_grad + (delta2.T).dot(a_1)
        Theta2_grad = Theta2_grad + (delta3.T).dot(a_2)

    
    
    Theta1_grad = Theta1_grad/m
    Theta2_grad = Theta2_grad/m
    
#Regularised
    reg_Theta1 = Theta1[:,1:]*(lambda_/m)
    reg_Theta2 = Theta2[:,1:]*(lambda_/m)
    
    
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + reg_Theta1 
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + reg_Theta2 
    
    
    
    

    
# Unroll gradients
#    grad = [Theta1_grad(:) ; Theta2_grad(:)];
    grad = np.hstack([Theta1_grad.T.ravel(), Theta2_grad.T.ravel()])
    


    return J, grad
    
def sigmoidGradient(z):
    return ((np.exp(-1*z)))/((1 + np.exp(-1*z))**2)
  
def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in)* 2 * epsilon_init - epsilon_init
    return W
    
def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    p=np.size(W)
    #q=W.shape
    return (np.sin(np.arange(p)+1)/10)
    #return (np.sin(np.arange(p)+1)/10).reshape((q),order='F')

def computeNumericalGradient(theta,in_size, hid_size, labels, X, y, lambda_):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    f = 1*10**(-4)
    
    for p in range(np.size(theta)):
# Set perturbation vector
        perturb[p] = f
        loss1, dummy = nnCostFunction(theta - perturb,in_size, hid_size, labels, X, y, lambda_)
        loss2, dummy = nnCostFunction(theta + perturb,in_size, hid_size, labels, X, y, lambda_)
# Compute Numerical Gradient
        #print('loss1 : {}'.format(loss1))
        #print('theta : {}'.format(theta[p]))
        #print('loss2 : {}'.format(loss2))
        numgrad[p] = (loss2 - loss1) / (2*f)
        #print('numgrad : {}'.format(numgrad[p]))
        perturb[p] = 0
    return numgrad
    
def checkNNGradients(*lambda_):
    if len(lambda_) <= 0:
        lambda_ = 0
    else:
        lambda_ = lambda_[0]
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size).reshape(hidden_layer_size,input_layer_size+1,order='F')
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size).reshape(num_labels,hidden_layer_size+1,order='F')
    #Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size).reshape(hidden_layer_size,input_layer_size+1)
    #Theta2 = debugInitializeWeights(num_labels, hidden_layer_size).reshape(num_labels,hidden_layer_size+1)


    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1).reshape(m,input_layer_size,order='F')
    y  = 1 + np.remainder((np.arange(m)+1), num_labels).T
    #print('X : {}'.format(X))
    #print('y : {}'.format(y))
    # Unroll parameters
      
    nn_params = np.hstack([Theta1.ravel(), Theta2.ravel()])
    nn_params1 = np.hstack([Theta1.T.ravel(), Theta2.T.ravel()])
    print('nn_params : {}'.format(nn_params1))  
    cost, grad = nnCostFunction1(nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)
    numgrad = computeNumericalGradient( nn_params1,input_layer_size, hidden_layer_size, num_labels, X, y, lambda_ );
    #print('grad : {}'.format(grad.shape))    
    #print('numgrad : {}'.format(numgrad.shape))
    print('{}'.format(np.vstack([numgrad,grad]).T))
    
    print('The above two columns you get should be very similar.\n','(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')
    
# Evaluate the norm of the difference between two solutions.  
# If you have a correct implementation, and assuming you used EPSILON = 0.0001 
# in computeNumericalGradient.m, then diff below should be less than 1e-9
    numgrad = numgrad.reshape((numgrad.shape[0],1))
    grad = grad.reshape((grad.shape[0],1))

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print(np.linalg.norm(numgrad-grad))
    print(np.linalg.norm(numgrad+grad))
    print('If your backpropagation implementation is correct, then \n the relative difference will be small (less than 1e-9). \nRelative Difference: {}\n'.format(diff))
    
    
    
########################parameters#######################################################
ratio = 1.00
lambda_ = 0
num_labels = 10

input_layer_size  = 400
hidden_layer_size = 25


      
my_file=r'C:\Users\dkashi200\Yawaris\machine-learning-ex4\ex4\ex4data1.mat'
test = sio.loadmat(my_file)
X1 = test['X']
y1 = test['y']

rand_indices = np.arange(test['X'].shape[0])
np.random.shuffle(rand_indices)
sel_x = X1[rand_indices][0:100,:]
sel_y = y1[rand_indices][0:100]
image_array_disp(sel_x)


m,n = X1.shape
m_train = int(m*ratio)
m_test = m - int(m*ratio)

X=X1[(rand_indices[0:m_train]),:]
y=y1[(rand_indices[0:m_train])]



my_file=r'C:\Users\dkashi200\Yawaris\machine-learning-ex4\ex4\ex4weights.mat'
test_NN = sio.loadmat(my_file)

Theta1=test_NN['Theta1']
Theta2=test_NN['Theta2']

nn_params = np.hstack([Theta1.ravel(), Theta2.ravel()])

print('Feedforward Using Neural Network ...\n')
J, grad=nnCostFunction1(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): {} \n(this value should be about 0.287629)\n'.format(J))

wait = input("PRESS ENTER TO CONTINUE.")

#% =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')

lambda_ = 1

J, grad=nnCostFunction1(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): {} \n(this value should be about 0.383770)\n'.format(J))

wait = input("PRESS ENTER TO CONTINUE.")


#% ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
# implement the gradient for the sigmoid function. You should complete the
# code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient(np.array([1, -0.5 ,0 ,0.5, 1]))

print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ')

print('{}'.format(g))

wait = input("PRESS ENTER TO CONTINUE.")

# ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)
print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.hstack([initial_Theta1.ravel() , initial_Theta2.ravel()])

wait = input("PRESS ENTER TO CONTINUE.")

#% =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#

print('\nChecking Backpropagation... \n')

checkNNGradients()

wait = input("PRESS ENTER TO CONTINUE.")

#% =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
lambda_ = 3
checkNNGradients(lambda_)

# Also output the costFunction debugging values

debug_J, dummy = nnCostFunction1(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lambda_)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = 3): {} \n(this value should be about 0.576051)\n\n'.format(debug_J))

wait = input("PRESS ENTER TO CONTINUE.")


# =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#

print('\nTraining Neural Network... \n')

# minimize the objective function
fmin = minimize(fun=nnCostFunction1, x0=nn_params, args=(input_layer_size, hidden_layer_size,num_labels, X, y, lambda_),  
                method='TNC', jac=True, options={'maxiter': 250})
fmin  

Theta1 = fmin['x'][0:(hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))

Theta2 = fmin['x'][(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,hidden_layer_size+1)


# ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
# the data.

print('\nVisualizing Neural Network... \n')

image_array_disp(Theta1[:,1:])

wait = input("PRESS ENTER TO CONTINUE.")

# ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict_nn(Theta1, Theta2, X)

print ('Train Accuracy:', (pred == y).mean() * 100)

X_test=X1[(rand_indices[m_train:]),:]
y_test=y1[(rand_indices[m_train:])]
pred = predict_nn(Theta1, Theta2, X_test)
print ('Test Accuracy:', (pred == y_test).mean() * 100)