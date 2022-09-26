#%%
# Importing libraries
# TODO: clean up imports
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import sklearn as skl
import numba as nb

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

#Lassso mit scikit-learn:
def lassoreg(X, z, lambdan):
    #Lasso regression with scikit-learn 
    RegLasso = Lasso(lambdan)
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

#TODO update to follow new standard (create X and send into func)
def cross_validation(k_deg_fold, x, y, z, model_fit=linreg, degree=2, lambdan=0):
    #step 1: shuffle datasets randomly using np.random.permutation(len(x)):
    assert len(x) == len(z) == len(y)
    p = np.random.permutation(len(x))
    x, y, z = x[p], y[p], z[p]
    
    #step 2: split the data in k groups with numpy.array_split
    x = np.array_split(x, k_deg_fold); 
    y = np.array_split(y, k_deg_fold); 
    z = np.array_split(z, k_deg_fold)

    # array to keep track of MSE for each test-group
    MSE_array = np.zeros((k_deg_fold))
    
    #step 3:
    for i in range(k_deg_fold):
        #a) pick one group to be test data
        x_test, y_test, z_test = x[i], y[i], z[i]
        
        #b) take remaining groupe as train data
        x_train = np.ndarray.flatten(np.array(x[:i] + x[i+1:]))
        y_train = np.ndarray.flatten(np.array(y[:i] + y[i+1:]))
        z_train = np.ndarray.flatten(np.array(z[:i] + z[i+1:]))
        
        #X_train = create_X_polynomial(x_train, y_train, 2)
        
        #c) fit model to train data
        if model_fit == linreg:
            beta = model_fit(x_train, y_train, z_train, degree)
        elif model_fit == ridgereg or model_fit == lassoreg:
            beta = model_fit(x_train, y_train, z_train, degree, lambdan)
        
        #d) evaluate model and save score-value
        X_test = create_X_polynomial(x_test, y_test, degree)
        z_tilde = X_test @ beta 
        MSE_array[i] = MSE(z_test, z_tilde)
        
    return MSE_array


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
    plt.show()
    

