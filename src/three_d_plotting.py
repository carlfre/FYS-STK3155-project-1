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


def plot_3d(X, Y, Z, title="", filename=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title, fontsize=20)
    if filename:
        plt.savefig(filename)
    plt.show()


def scatter_3d(x, y, z, title="", filename=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc_plot = ax.scatter(x, y, z, c=z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.colorbar(sc_plot)
    plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_Franke(filename=""):
    """Plot the Franke function on [0,1] x [0,1] """
    M = 100
    x = np.linspace(0, 1, M)
    y = np.linspace(0, 1, M)
    X, Y = np.meshgrid(x, y)
    Z = franke_function(X, Y)

    plot_3d(X, Y, Z, title="Franke function", filename=filename)


def plot_estimated_environment(model_type="ols", n=5, lmbda=None, save=False):
    from sklearn.preprocessing import StandardScaler
    
    # estimate beta
    seed = 38748
    N = 400
    sigma2 = 0.1

    x, y, z, _ = generate_data_Franke(N, sigma2, seed)
    #z_mean = np.mean(z)
    #z -= 

    # Use linear regression
    model = rt.LinearRegression(model_type, lmbda)

    # Fit an n-degree polynomial
    X = rt.create_X_polynomial(x, y, n)

    #scaler = StandardScaler()
    #scaler.fit(X)
    #X = scaler.transform(X)

    beta = model(X, z)

    # create a grid of points to evaluate the model at
    M = 100
    xstar = np.linspace(0, 1, M)
    ystar = np.linspace(0, 1, M)

    X, Y = np.meshgrid(xstar, ystar)
    ny, nx = X.shape
    X_star = X.flatten()
    Y_star = Y.flatten()
    design = rt.create_X_polynomial(X_star, Y_star, n)

    #design = scaler.transform(design)

    Z_star = design @ beta
    Z = Z_star.reshape((ny, nx))


    # Plot the data in 3D
    title = f"{str(model)}, n={n}"
    if lmbda is not None:
        title += f", $\lambda$={lmbda:e}"

    if save:
        filename=f"figures/3d_{model_type}_{n}.pdf"
    else:
        filename=""
    plot_3d(X, Y, Z, title=title, filename=filename)


def main():
    #plot_estimated_environment("ols", 5,)
    #plot_estimated_environment("ridge", 5, lmbda=1e-10)
    #plot_estimated_environment("lasso", 5, lmbda=1e-5)
    plot_Franke()


if __name__ == "__main__":
    main()
# %%
