import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler


import regression_tools as rt
from analysis import generate_data_Franke
from three_d_plotting import scatter_3d


def plot_data(x, y, z, title=""):
    scatter_3d(x, y, z, title=title)


def OLS_evalation(x, y, z):
    ols = rt.LinearRegression("ols")
    # Evaluate R2 and MSE for polynomials of degree up to 5
    degreerange = 20
    degrees = range(2, degreerange + 1)
    MSE_arr, R2_arr, beta_arr = [], [], []
    MSE_arr_train = []
    R2_arr_train = []

    # Saving R2- and MSE-values in arrays for polynomials of degree up to 5
    for deg in degrees:
        X = rt.create_X_polynomial(x, y, deg)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        z_s = scaler.fit_transform(z.reshape(-1, 1)).flatten()
        X_train_s, X_test_s, z_train_s, z_test_s = train_test_split(X_s,z_s,test_size=0.25)
        beta = ols(X_train_s, z_train_s)
        beta_arr.append(beta)
        ztilde_test = X_test_s @ beta
        ztilde_train = X_train_s @ beta

        R2_arr.append(rt.R2(z_test_s, ztilde_test))
        R2_arr_train.append(rt.R2(z_train_s, ztilde_train))
        MSE_arr.append(rt.MSE(z_test_s, ztilde_test))
        MSE_arr_train.append(rt.MSE(z_train_s, ztilde_train))


    # Plotting MSE and R2 for all polynomial orders between 2 and 5
    plt.plot(degrees, MSE_arr, "--o", label="MSE - Test")
    plt.plot(degrees, MSE_arr_train, "--o", label="MSE - Train")
    plt.xlabel("Polynomial order", fontsize=16)
    plt.ylabel("Mean Square Error", fontsize=16)
    plt.title("Change in MSE with increasing polynomial order", fontsize=18)
    plt.xticks(ticks=degrees,fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.show()

    plt.plot(degrees, R2_arr, "--o", label="R2 - Test")
    plt.plot(degrees, R2_arr_train, "--o", label= "R2 - train")
    plt.xlabel("Polynomial order", fontsize=16)
    plt.ylabel("R2-score", fontsize=16)
    plt.title("Change in R2 with increasing polynomial order", fontsize=18)
    plt.xticks(ticks=degrees,fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Plotting first five parameters in beta for all polynomial orders between 2 and 5
    plt.figure(figsize=(5, 5)) 
    for i in range(5):
        plt.plot(degrees, [b[i] for b in beta_arr], "--o",  label=fr"Coefficient $\beta_{i}$")
    plt.title("First five parameters in beta for increasing polynomial order", fontsize=18)
    plt.xlabel("Polynomial order", fontsize=16)
    plt.ylabel("Coefficient value", fontsize=16)
    plt.xticks(ticks=degrees,fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def OLS_bias_variance(x, y, z):
    print("Problem c)")

    B = 100 # n_bootsraps 

    # Initialize model
    ols = rt.LinearRegression("ols")

    # Plot train & test MSE for degree up to 10
    degreerange = 20
    degrees = range(1, degreerange + 1)


    # Bootstrap for bias-variance tradeoff analysis
    degrees = list(range(1, degreerange + 1))
    biases, variances, errors = [], [], []
    for deg in degrees:
        X = rt.create_X_polynomial(x, y, deg)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        z_s = scaler.fit_transform(z.reshape(-1, 1)).flatten()
        bias, variance, error = rt.bootstrap(X_s, z_s, B, ols)
        biases.append(bias)
        variances.append(variance)
        errors.append(error)

    plt.plot(degrees, biases, "--o" , label="Bias$^2$")
    plt.plot(degrees, variances, "--o" ,label="Variance")
    plt.plot(degrees, errors, "--o", label="Error")
    plt.title("Bias, variance and error for polynomial degrees", fontsize=18)
    plt.xlabel("Polynomial order", fontsize=16)
    plt.ylabel("Error scores", fontsize=16)
    plt.xticks(ticks=degrees,fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def find_lambda_range(x, y, z):

    #Plot train & test MSE for degree up to 10 with lambda l
    degree = 20
    k_fold = 5
    lambda_range = np.logspace(-10, 10, 100)
    MSE_train_ridge = []
    MSE_test_ridge = []
    MSE_train_lasso = []
    MSE_test_lasso = []

    for l in lambda_range:
        print(l)
        X = rt.create_X_polynomial(x, y, degree)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        z_s = scaler.fit_transform(z.reshape(-1, 1)).flatten()
        ridge = rt.LinearRegression("ridge", l)
        MSECV_train, MSECV_test = rt.cross_validation(X_s, z_s, k_fold, ridge)
        MSE_train_ridge.append(np.mean(MSECV_train))
        MSE_test_ridge.append(np.mean(MSECV_test))
        lasso = rt.LinearRegression("lasso", l)
        MSECV_train, MSECV_test = rt.cross_validation(X_s, z_s, k_fold, lasso)
        MSE_train_lasso.append(np.mean(MSECV_train))
        MSE_test_lasso.append(np.mean(MSECV_test))

    # Plotting MSE for all polynomial orders between 2 and 10
    plt.plot(np.log10(lambda_range), MSE_train_ridge, label="Train data MSE ridge")
    plt.plot(np.log10(lambda_range), MSE_test_ridge, label="Test data MSE ridge")
    plt.plot(np.log10(lambda_range), MSE_train_lasso, label="Train data MSE lasso")
    plt.plot(np.log10(lambda_range), MSE_test_lasso, label="Test data MSE lasso")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(fr"$\log (\lambda)$", fontsize=16)
    plt.ylabel("Mean Square Error", fontsize=16)
    plt.title(fr"MSE as a function of $\log(\lambda)$ for degree {degree}",fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show() 


def plot_MSE_comp(x, y, z):
    # set k-fold
    k_fold = 5

    # set a labda ranges:
    lambda_range_lasso = np.logspace(-4, -1)
    lambda_range_ridge = np.logspace(-10, 2)

    # set degree range of interest
    deg_range = np.array(range(2, 20 + 1))
    lambda_optimal_lasso = np.zeros(len(deg_range))
    lambda_optimal_ridge = np.zeros(len(deg_range))

    MSE_ols = []
    MSE_ridge = []
    MSE_lasso = []

    # for each degree we want to calculate the parameter lambda, within our range
    # which optimize the ridge regression
    for i, deg in enumerate(deg_range):
        print(deg)
        X = rt.create_X_polynomial(x, y, deg)
        scaling = StandardScaler()
        X_s = scaling.fit_transform(X)
        z_s = scaling.fit_transform(z.reshape(-1, 1)).flatten()
        lambda_optimal_lasso[i] = rt.CV_gridsearch(X_s, z_s, k_fold, lambda_range_lasso, "ridge")
        lambda_optimal_ridge[i] = rt.CV_gridsearch(X_s, z_s, k_fold, lambda_range_ridge, "ridge")
        
        X_train_s, X_test_s, z_train_s, z_test_s = train_test_split(X_s, z_s, test_size=0.25)
        
        ols = rt.LinearRegression("ols")
        beta_ols = ols(X_train_s, z_train_s)
        ztilde_ols_s = X_test_s@beta_ols

        lasso = rt.LinearRegression("lasso", lambda_optimal_lasso[i])
        beta_lasso = lasso(X_train_s, z_train_s)
        ztilde_lasso_s = X_test_s@beta_lasso

        ridge = rt.LinearRegression("ridge", lambda_optimal_ridge[i])
        beta_ridge = ridge(X_train_s, z_train_s)
        ztilde_ridge_s = X_test_s@beta_ridge

        MSE_ols.append(rt.MSE(z_test_s, ztilde_ols_s))
        MSE_lasso.append(rt.MSE(z_test_s, ztilde_lasso_s))
        MSE_ridge.append(rt.MSE(z_test_s, ztilde_ridge_s))

    plt.plot(deg_range, np.log10(lambda_optimal_ridge), "--o", c="y")
    plt.ylabel(r"$\log(\lambda)$", fontsize=18)
    plt.xlabel("Polynomial order", fontsize=18)
    plt.xticks(deg_range, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(r"Optimal $\lambda$ in ridge for each polonomial order" "\n" r"Scope of $\log(\lambda)$ between [-10, 2]", fontsize=18)
    plt.tight_layout()
    plt.show()

    plt.plot(deg_range, np.log10(lambda_optimal_lasso), "--o", c="r")
    plt.ylabel(r"$\log(\lambda)$", fontsize=18)
    plt.xlabel("Polynomial order", fontsize=18)
    plt.xticks(deg_range, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(r"Optimal $\lambda$ in lasso for each polonomial order" "\n" r"Scope of $\log(\lambda)$ between [-4, -1]", fontsize=18)
    plt.tight_layout()
    plt.show()

    plt.scatter(deg_range[np.argmin(MSE_ols)], np.min(MSE_ols), c="b",  label="Optimal model ols", zorder=6)
    plt.scatter(deg_range[np.argmin(MSE_ridge)], np.min(MSE_ridge), c="g", label="Optimal model ridge", zorder=5)
    plt.scatter(deg_range[np.argmin(MSE_lasso)], np.min(MSE_lasso), c="k", label="Optimal model lasso", zorder=4)
    plt.plot(deg_range, MSE_ols, "--o", label="ols",zorder=3)
    plt.plot(deg_range, MSE_ridge, "--o", label="ridge", zorder=2)
    plt.plot(deg_range, MSE_lasso, "--o", label="lasso", zorder=1)
    plt.xlabel("Polynomal order", fontsize=18)
    plt.ylabel("Mean Square Error", fontsize=18)
    plt.xticks(deg_range, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(fr"Comparison of MSE between models for optimized $\lambda$", fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    #create data
    seed = 199
    N = 1000
    sigma2 = 0.1 # Variance of noise
    x, y, z, _ = generate_data_Franke(N, sigma2, seed)

    #scatter_3d(x, y, z, title=fr"Sampled data, $\sigma^2=0.1$")
    plot_MSE_comp(x, y, z)

if __name__ == "__main__":
    main()