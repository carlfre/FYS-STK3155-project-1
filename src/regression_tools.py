#%%
# Importing libraries
# TODO: clean up imports
from copy import copy
from re import L
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import sklearn as skl
#import numba as nb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Defining functions

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def create_X_polynomial(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)      # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def linreg(X, z):
    # Solving for beta
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z
    return beta

#Ridge regression
def ridgereg(X, z, lambdan):
    #create identity matrix
    I = np.eye(len(X.T), len(X.T))

    # Solving for beta
    beta_ridge = np.linalg.pinv(X.T@X + lambdan*I) @ X.T @ z
    return beta_ridge

#Lassso with scikit-learn:
def lassoreg(X, z, lambdan):
    #Lasso regression with scikit-learn 
    RegLasso = Lasso(lambdan, fit_intercept = False)
    RegLasso.fit(X,z)
    beta_lasso = RegLasso.coef_
    return beta_lasso

def bootstrap_linreg(X, z, B):
    """Returns estimated distributions of beta estimators."""
    t = np.zeros(B)
    n_datapoints = len(z)
    
    beta = linreg(X, z)
    distribution = np.zeros((len(beta), B))
    for b in range(B):
        datapoints = np.random.randint(0,n_datapoints,n_datapoints)
        X_b = X[datapoints]
        z_b = z[datapoints]
        beta_b = linreg(X_b, z_b)
        distribution[:, b] = beta_b
    return distribution

def CV_linreg(k_deg_fold, X, z):
    "Preformes Cross-Validation with model=linreg, X=design-matrix"
    "Returns arrays - MSE-train, MSE-test - with length k_deg_fold"
    "np.mean() on output is estimated MSE with cross-validation"

    #step 1: shuffle datasets randomly using np.random.permutation(len(x)):
    assert len(X) == len(z)
    p = np.random.permutation(len(X))
    X, z = X[p], z[p]
    
    #step 2: split the data in k groups with numpy.array_split
    X = np.array_split(X, k_deg_fold) 
    z = np.array_split(z, k_deg_fold)

    #array to keep track of MSE for each test-group and train-group
    MSE_train = np.zeros((k_deg_fold))
    MSE_test = np.zeros((k_deg_fold))

    #step 3: for i in range of folds preform:
    for i in range(k_deg_fold):
        #a) pick one group to be test data
        X_test , z_test = X[i], z[i]
        
        #b) take remaining groups as train data
            #np.delete() creates a "new" array (does not alter X)
            #concatenate merges remaining groups to train data
        X_train = np.concatenate([m for m in np.delete(X, i, axis=0)])
        z_train = np.concatenate([m for m in np.delete(z, i, axis=0)])
        
        #c) fit model to train data with linreg and compute z_tilde 
        beta = linreg(X_train, z_train)
        z_tilde_test = X_test @ beta 
        z_tilde_train = X_train @ beta

        #d) evaluate model and save score-value to MSE-arrays
        MSE_train[i] = MSE(z_train, z_tilde_train)
        MSE_test[i] = MSE(z_test, z_tilde_test)
    
    return MSE_train, MSE_test

def CV_lassoreg(k_deg_fold, X, z, lambdan):
    "Preformes Cross-Validation with model=lassoreg, X=design-matrix"
    "Returns arrays - MSE-train, MSE-test - with length k_deg_fold"
    "np.mean() on output is estimated MSE with cross-validation"

    #step 1: shuffle datasets randomly using np.random.permutation(len(x)):
    assert len(X) == len(z)
    p = np.random.permutation(len(X))
    X, z = X[p], z[p]
    
    #step 2: split the data in k groups with numpy.array_split
    X = np.array_split(X, k_deg_fold) 
    z = np.array_split(z, k_deg_fold)

    #array to keep track of MSE for each test-group and train-group
    MSE_train = np.zeros((k_deg_fold))
    MSE_test = np.zeros((k_deg_fold))

    #step 3: for i in range of folds preform:
    for i in range(k_deg_fold):
        #a) pick one group to be test data
        X_test , z_test = X[i], z[i]
        
        #b) take remaining groups as train data
            #np.delete() creates a "new" array (does not alter X)
            #concatenate merges remaining groups to train data
        X_train = np.concatenate([m for m in np.delete(X, i, axis=0)])
        z_train = np.concatenate([m for m in np.delete(z, i, axis=0)])

        #c) fit model to train data with lasso-reggresion 
        beta = lassoreg(X_train, z_train, lambdan)

        #d) evaluate model and save score-value
        z_tilde_test = X_test @ beta 
        z_tilde_train = X_train @ beta

        MSE_train[i] = MSE(z_train, z_tilde_train)
        MSE_test[i] = MSE(z_test, z_tilde_test)

    return MSE_train, MSE_test

def CV_ridgereg(k_deg_fold, X, z, lambdan):
    "Preformes Cross-Validation with model=ridgereg, X=design-matrix"
    "Returns arrays - MSE-train, MSE-test - with length k_deg_fold"
    "np.mean() on output is estimated MSE with cross-validation"

    #step 1: shuffle datasets randomly using np.random.permutation(len(x)):
    assert len(X) == len(z)
    p = np.random.permutation(len(X))
    X, z = X[p], z[p]
    
    #step 2: split the data in k groups with numpy.array_split
    X = np.array_split(X, k_deg_fold) 
    z = np.array_split(z, k_deg_fold)

    #array to keep track of MSE for each test-group and train-group
    MSE_train = np.zeros((k_deg_fold))
    MSE_test = np.zeros((k_deg_fold))

    #step 3: for i in range of folds preform:
    for i in range(k_deg_fold):
        #a) pick one group to be test data
        X_test , z_test = X[i], z[i]
        
        #b) take remaining groups as train data
            #np.delete() creates a "new" array (does not alter X)
            #concatenate merges remaining groups to train data
        X_train = np.concatenate([m for m in np.delete(X, i, axis=0)])
        z_train = np.concatenate([m for m in np.delete(z, i, axis=0)])
        
        #c) fit model to train data with ridge-reggresion
        beta = ridgereg(X_train, z_train, lambdan)

        #d) evaluate model and save score-value
        z_tilde_test = X_test @ beta
        z_tilde_train = X_train @ beta

        MSE_train[i] = MSE(z_train, z_tilde_train)
        MSE_test[i] = MSE(z_test, z_tilde_test)  

    return MSE_train, MSE_test

def bootstrap_ridge(X, z, B, lambdan):
    """Returns estimated distributions of beta estimators."""
    n_datapoints = len(z)
    
    beta = ridgereg(X, z, lambdan)
    distribution = np.zeros((len(beta), B))
    for b in range(B):
        datapoints = np.random.randint(0,n_datapoints,n_datapoints)
        X_b = X[datapoints]
        z_b = z[datapoints]
        beta_b = ridgereg(X_b, z_b, lambdan)
        distribution[:, b] = beta_b
    return distribution

def bootstrap_lasso(X, z, B, lambdan):
    """Returns estimated distributions of beta estimators."""
    n_datapoints = len(z)
    
    beta = lassoreg(X, z, lambdan)
    distribution = np.zeros((len(beta), B))
    for b in range(B):
        datapoints = np.random.randint(0,n_datapoints,n_datapoints)
        X_b = X[datapoints]
        z_b = z[datapoints]
        beta_b = lassoreg(X_b, z_b, lambdan)
        distribution[:, b] = beta_b
    return distribution

if __name__ == "__main__":
    # Generating data
    N = 1000
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    z = FrankeFunction(x, y)

    # Fit an n-degree polynomial
    n = 5
    X = create_X_polynomial(x, y, n)
    beta = linreg(X, z)
    ztilde = X @ beta

    
    B = 100
    distribution = bootstrap_linreg(X, z, B)

    # Plots estimated distribution for the i'th parameter of the model
    i = -1
    plt.hist(distribution[i, :])
    #plt.show()

    k_fold = 8
    lambdan = 0.0001
    MSE_arr = bootstrap_lasso(X, z, 100, lambdan)
    print(MSE_arr)

    

