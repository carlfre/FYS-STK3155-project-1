#%%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

import regression_tools as rt
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def problem_b():
    print("Problem b)")

    np.random.seed(199)

    # Generating data
    N = 1000
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    sigma = 1
    epsilon = np.random.normal(0, sigma)  # Random noise to add to func
    z = rt.FrankeFunction(x, y) + epsilon

    # Create design matrix and train-test-split
    deg = 5 # degree of polynomial
    X = rt.create_X_polynomial(x, y, deg)
    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.25)

    # Estimate beta
    beta = rt.linreg(X_train, z_train)

    # Evaluate MSE and R2
    ztilde_train = X_train @ beta
    ztilde_test = X_test @ beta

    print(f"Linear regression with polynomial of degree {deg}:")
    print("R2 - training")
    print(rt.R2(z_train, ztilde_train))
    print("MSE - training")
    print(rt.MSE(z_train, ztilde_train))
    print()
    print("R2 - testing")
    print(rt.R2(z_test, ztilde_test))
    print("MSE - testing")
    print(rt.MSE(z_test, ztilde_test))
    print()

    # Evaluate R2 and MSE for polynomials of degree up to 5
    degreerange = 5
    degrees = range(2, degreerange + 1)
    MSE_arr, R2_arr, beta_arr = [], [], []

    # Saving R2- and MSE-values in arrays for polynomials of degree up to 5
    for deg in degrees:
        X = rt.create_X_polynomial(x, y, deg)
        X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.25)
        beta = rt.linreg(X_train, z_train)
        beta_arr.append(beta)
        ztilde_test = X_test @ beta

        R2_arr.append(rt.R2(z_test, ztilde_test))
        
        MSE_arr.append(rt.MSE(z_test, ztilde_test))

    # Plotting MSE and R2 for all polynomial orders between 2 and 5
    plt.plot(degrees, MSE_arr)
    plt.xlabel("Polynomial order")
    plt.ylabel("Mean Square Error")
    plt.title("Change in MSE with increasing polynomial order")
    plt.savefig("../plots/Oppgave_b/MSE.pdf")
    plt.show()
    plt.clf()
    plt.plot(degrees, R2_arr)
    plt.xlabel("Polynomial order")
    plt.ylabel("R2-score")
    plt.title("Change in R2 with increasing polynomial order")
    plt.savefig("../plots/Oppgave_b/R2.pdf")
    plt.show()
    print("Generated MSE and R2 plots.")
    
    # Plotting first five parameters in beta for all polynomial orders between 2 and 5
    plt.figure(figsize=(5, 5)) 
    for i in range(degreerange):
        plt.plot(degrees, [b[i] for b in beta_arr], label=fr"Coefficient $\beta_{i}$")
    plt.title("First four parameters in beta for increasing polynomial order")
    plt.xlabel("Polynomial order")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.savefig("../plots/Oppgave_b/beta.pdf")
    plt.show()
    print("Generated parameter value plot.")



def problem_c():
    print("Problem c)")
    np.random.seed(569)

    # Generating data
    N = 100
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    
    sigma = 0.1
    epsilon = np.random.normal(0, sigma)
    z = rt.FrankeFunction(x, y) + epsilon

    # Plot train & test MSE for degree up to 10
    degreerange = 10
    degrees = range(1, degreerange + 1)
    MSE_train = []
    MSE_test = []

    for deg in degrees:
        X = rt.create_X_polynomial(x, y, deg)
        X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.25)
        beta = rt.linreg(X_train, z_train)
        ztilde_train = X_train @ beta
        ztilde_test = X_test @ beta
        
        MSE_train.append(rt.MSE(z_train, ztilde_train))
        MSE_test.append(rt.MSE(z_test, ztilde_test))

    # Plotting MSE and R2 for all polynomial orders between 2 and 5
    plt.plot(degrees, MSE_train, label="Train data MSE")
    plt.plot(degrees, MSE_test, label="Test data MSE")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Mean Square Error")
    plt.title("Train and test MSE as a function of model complexity")
    plt.legend()
    plt.savefig("../plots/Oppgave_c/train_v_test_MSE.pdf")
    plt.show()
    plt.clf()
    print("Generated train v test MSE plot")

    # Bootstrap for bias-variance tradeoff analysis
    B = 100
    degrees = range(1, 11)
    errors, biases, variances = [], [], []
    for deg in degrees:
        X = rt.create_X_polynomial(x, y, deg)
        X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.25)
        distribution = rt.bootstrap_linreg(X_train, z_train, B)
        z_pred = X_test @ distribution
        error = np.mean( np.mean((z_test.reshape(-1, 1) - z_pred)**2, axis=1, keepdims=True) )
        bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )
        errors.append(error)
        biases.append(bias)
        variances.append(variance)
    plt.plot(degrees, errors, label="Error")
    plt.plot(degrees, biases, label="$Bias^2$")
    plt.plot(degrees, variances, label="Variance")
    plt.savefig("../plots/Oppgave_c/bias_variance_tradeoff.pdf")
    plt.show()
    plt.clf()



def problem_d():
    print("Problem d)")
    np.random.seed(569)

    # Generating data
    N = 150
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    
    sigma = 1
    epsilon = np.random.normal(0, sigma)  # Random noise to add to func
    z = rt.FrankeFunction(x, y) + epsilon

    # Plot train & test MSE for degree up to 10
    k_fold_num = 5
    degreerange = 10
    degrees = range(1, degreerange + 1)
    MSE_train = []
    MSE_test = []

    for deg in degrees:
        X = rt.create_X_polynomial(x, y, deg)
        MSECV_train, MSECV_test = rt.CV_linreg(k_fold_num, X, z)
        MSE_train.append(np.mean(MSECV_train))
        MSE_test.append(np.mean(MSECV_test))

    # Plotting MSE for all polynomial orders between 2 and 10 with CV
    plt.plot(degrees, MSE_train, label="Train data MSE")
    plt.plot(degrees, MSE_test, label="Test data MSE")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Mean Square Error")
    plt.title(f"Cross validation MSE as a function of polynomial degrees with N={N}")
    plt.legend()
    plt.savefig(f"../plots/Oppgave_d/Cross_Validation_MSE_N={N}.pdf")
    plt.show()
    plt.clf()
    print("Generated train v test MSE plot")


def problem_e():
    print("Problem e)")
    #Preform task c-d with ridge-regression for different lambdas
    lambdas = np.array([0.00001, 0.001, 0.1, 10])
    np.random.seed(569)

    # Generating data
    N = 100
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    
    sigma = 1
    epsilon = np.random.normal(0, sigma)  # Add random noise to func
    z = rt.FrankeFunction(x, y) + epsilon

    for l in lambdas:
        # Plot train & test MSE for degree up to 10
        degreerange = 10
        degrees = range(1, degreerange + 1)
        MSE_train = []
        MSE_test = []

        for deg in degrees:
            X = rt.create_X_polynomial(x, y, deg)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.25)
            beta = rt.ridgereg(X_train, z_train, l)
            ztilde_train = X_train @ beta
            ztilde_test = X_test @ beta
            
            MSE_train.append(rt.MSE(z_train, ztilde_train))
            MSE_test.append(rt.MSE(z_test, ztilde_test))

        # Plotting MSE for all polynomial orders between 2 and 10
        plt.plot(degrees, MSE_train, label="Train data MSE")
        plt.plot(degrees, MSE_test, label="Test data MSE")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Mean Square Error")
        plt.title(f"Ridge regression MSE as a function of polynomial degrees with lambda = {l}")
        plt.legend()
        plt.savefig(f"../plots/Oppgave_e/Ridge_test_train_lambda_{l}.pdf")
        plt.show()
        print("Generated train v test MSE plot")

        # Bootstrap for bias-variance tradeoff analysis
        B = 100
        degrees = range(1, 11)
        errors, biases, variances = [], [], []
        for deg in degrees:
            X = rt.create_X_polynomial(x, y, deg)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.25)
            distribution = rt.bootstrap_ridge(X_train, z_train, B, l)
            z_pred = X_test @ distribution
            error = np.mean( np.mean((z_test.reshape(-1, 1) - z_pred)**2, axis=1, keepdims=True) )
            bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )
            errors.append(error)
            biases.append(bias)
            variances.append(variance)
        plt.plot(degrees, errors, label="Error")
        plt.plot(degrees, biases, label="$Bias^2$")
        plt.plot(degrees, variances, label="Variance")
        plt.title(f"Error, Bias and Variances as function of polynomial degrees for lambda = {l}")
        plt.legend()
        plt.savefig(f"../plots/Oppgave_e/Ridge_error_bias_variance_lambda_{l}.pdf")
        plt.show()

        # Plot train & test MSE for degree up to 10 with lambda l
        k_fold_num = 5
        degreerange = 10
        degrees = range(1, degreerange + 1)
        MSE_train = []
        MSE_test = []

        for deg in degrees:
            X = rt.create_X_polynomial(x, y, deg)
            MSECV_train, MSECV_test = rt.CV_ridgereg(k_fold_num, X, z, l)
            MSE_train.append(np.mean(MSECV_train))
            MSE_test.append(np.mean(MSECV_test))

        # Plotting MSE for all polynomial orders between 2 and 10
        plt.plot(degrees, MSE_train, label="Train data MSE")
        plt.plot(degrees, MSE_test, label="Test data MSE")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Mean Square Error")
        plt.title(f"Cross validation MSE with ridge regression \n as a function of polynomial degrees with lambda {l}")
        plt.legend()
        plt.savefig(f"../plots/Oppgave_e/Ridge_Cross_Validation_MSE_lambda_{l}.pdf")
        plt.show()
        plt.clf()
        print("Generated train v test MSE plot")   


def problem_f():
    print("Problem f)")
    #Preform task c-d with lasso-regression for different lambdas
    lambdas = np.array([0.00001, 0.001, 0.1, 10])
    
    np.random.seed(569)

    # Generating data
    N = 1000
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    
    sigma = 1
    epsilon = np.random.normal(0, sigma)  # Add random noise to func
    z = rt.FrankeFunction(x, y) + epsilon

    for l in lambdas:
        # Plot train & test MSE for degree up to 10
        degreerange = 10
        degrees = range(1, degreerange + 1)
        MSE_train = []
        MSE_test = []

        for deg in degrees:
            X = rt.create_X_polynomial(x, y, deg)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.25)
            beta = rt.lassoreg(X_train, z_train, l)
            ztilde_train = X_train @ beta
            ztilde_test = X_test @ beta
            
            MSE_train.append(rt.MSE(z_train, ztilde_train))
            MSE_test.append(rt.MSE(z_test, ztilde_test))

        # Plotting MSE for all polynomial orders between 2 and 10
        plt.plot(degrees, MSE_train, label="Train data MSE")
        plt.plot(degrees, MSE_test, label="Test data MSE")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Mean Square Error")
        plt.title(f"Lasso regression MSE as a function of polynomial degrees for lambda = {l}")
        plt.legend()
        plt.savefig(f"../plots/Oppgave_f/Lasso_test_train_lambda_{l}.pdf")
        plt.show()
        print("Generated train v test MSE plot")

        # Bootstrap for bias-variance tradeoff analysis
        B = 100
        degrees = range(1, 11)
        errors, biases, variances = [], [], []
        for deg in degrees:
            X = rt.create_X_polynomial(x, y, deg)
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.25)
            distribution = rt.bootstrap_lasso(X_train, z_train, B, l)
            z_pred = X_test @ distribution
            error = np.mean( np.mean((z_test.reshape(-1, 1) - z_pred)**2, axis=1, keepdims=True) )
            bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )
            errors.append(error)
            biases.append(bias)
            variances.append(variance)
        plt.plot(degrees, errors, label="Error")
        plt.plot(degrees, biases, label="$Bias^2$")
        plt.plot(degrees, variances, label="Variance")
        plt.title(f"Error, Bias and Variances as function of model complexity for lambda = {l}")
        plt.legend()
        plt.savefig(f"../plots/Oppgave_f/Lasso_error_bias_variance_lambda_{l}.pdf")
        plt.show()

        # Plot train & test MSE for degree up to 10
        k_fold_num = 5
        degreerange = 10
        degrees = range(1, degreerange + 1)
        MSE_train = []
        MSE_test = []

        for deg in degrees:
            X = rt.create_X_polynomial(x, y, deg)
            MSECV_train, MSECV_test = rt.CV_lassoreg(k_fold_num, X, z, l)
            MSE_train.append(np.mean(MSECV_train))
            MSE_test.append(np.mean(MSECV_test))

        # Plotting MSE for all polynomial orders between 2 and 10
        plt.plot(degrees, MSE_train, label="Train data MSE")
        plt.plot(degrees, MSE_test, label="Test data MSE")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Mean Square Error")
        plt.title(f"Cross validation MSE with lasso regression \n as a function of polynomial degrees with lambda {l}")
        plt.legend()
        plt.savefig(f"../plots/Oppgave_f/Lasso_Cross_Validation_MSE_lambda_{l}.pdf")
        plt.show()
        plt.clf()
        print("Generated train v test MSE plot") 


def problem_g():
    """Test on real data"""
    # Load the terrain
    terrain1 = imread("SRTM_Saarland.tif")
    # Show the terrain
    plt.figure()
    plt.title("Terrain over Saarland")
    plt.imshow(terrain1, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    

def main():
    problem_f()

if __name__ == "__main__":
    main()
# %%
