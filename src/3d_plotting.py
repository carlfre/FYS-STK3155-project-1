#%%
import numpy as np
import matplotlib.pyplot as plt

import regression_tools as rt
from analysis import generate_data_Franke, franke_function



def scatter_plot():
    seed = 898
    N = 400
    sigma2 = 0.1

    x, y, z, _ = generate_data_Franke(N, sigma2, seed)

    # Use linear regression
    ols = rt.LinearRegression("ols")

    # Fit an n-degree polynomial
    n = 5
    X = rt.create_X_polynomial(x, y, n)
    beta = ols(X, z)

    ztilde = X @ beta


    # Plot the data in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b')
    ax.scatter(x, y, ztilde, c='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_Franke():
    """Plot the Franke function on [0,1] x [0,1] """
    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, 1000)
    X, Y = np.meshgrid(x, y)
    Z = franke_function(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def evaluate_at_xy(x, y, beta, n):
    X = rt.create_X_polynomial(x, y, n)
    return X @ beta


def plot_estimated_environment():
    # estimate beta
    seed = 522
    N = 400
    sigma2 = 0.1

    x, y, z, _ = generate_data_Franke(N, sigma2, seed)

    # Use linear regression
    ols = rt.LinearRegression("ols")

    # Fit an n-degree polynomial
    n = 5
    X = rt.create_X_polynomial(x, y, n)
    beta = ols(X, z)

    # create a grid of points to evaluate the model at
    xstar = np.linspace(0, 1, 100)
    ystar = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xstar, ystar)
    Z = evaluate_at_xy(X, Y, beta, n)
    # Plot the data in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()





def main():
    plot_estimated_environment()
    #plot_Franke()
    #scatter_plot()


if __name__ == "__main__":
    main()