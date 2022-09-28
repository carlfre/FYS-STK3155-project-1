#%%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

import regression_tools as rt



def problem_b():
    print("Problem b)")

    np.random.seed(199)

    # Generating data
    N = 1000
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    sigma= 0.1
    epsilon = np.random.normal(0, sigma)
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

    # Plotting fits for all polynomial orders between 2 and 5
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
    plt.savefig("plots/MSE.pdf")
    plt.show()
    plt.clf()
    plt.plot(degrees, R2_arr)
    plt.xlabel("Polynomial order")
    plt.ylabel("R2-score")
    plt.title("Change in R2 with increasing polynomial order")
    plt.savefig("plots/R2.pdf")
    plt.show()
    plt.clf()
    print("Generated MSE and R2 plots.")
    
    # Plotting parameters in beta for all polynomial orders between 2 and 5
    plt.figure(figsize=(5, 5)) 
    for i in range(degreerange):
        plt.plot(degrees, [b[i] for b in beta_arr], label=f"Coefficient {i}")
    plt.title("First four parameters in beta for increasing polynomial order")
    plt.xlabel("Polynomial order")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.savefig("plots/beta.pdf")
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

    # Plot train & test MSE for degree up to 20
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
    plt.savefig("plots/train_v_test_MSE.pdf")
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
    plt.savefig("plots/bias_variance_tradeoff.pdf")
    plt.show()
    plt.clf()


def problem_d():
    print("Problem d)")
    np.random.seed(569)

    # Generating data
    N = 150
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    
    sigma = 0.1
    epsilon = np.random.normal(0, sigma)
    z = rt.FrankeFunction(x, y) + epsilon

    # Plot train & test MSE for degree up to 20
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

    # Plotting MSE and R2 for all polynomial orders between 2 and 5
    plt.plot(degrees, MSE_train, label="Train data MSE")
    plt.plot(degrees, MSE_test, label="Test data MSE")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Mean Square Error")
    plt.title("Train and test MSE as a function of model complexity - CV")
    plt.legend()
    plt.savefig("plots/Cross_Validation_MSE.pdf")
    plt.show()
    plt.clf()
    print("Generated train v test MSE plot")

#ToDo clean up (titles, labels etc.)
def problem_e():
    print("Problem e)")
    #Preform c-d whith ridge-reggresion for different lambda
    lambdas = np.array([0.00001, 0.001, 0.1, 10])
    
    np.random.seed(569)

    # Generating data
    N = 100
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    
    sigma = 0.1
    epsilon = np.random.normal(0, sigma)
    z = rt.FrankeFunction(x, y) + epsilon

    for l in lambdas:
        # Plot train & test MSE for degree up to 20
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

        # Plotting MSE and R2 for all polynomial orders between 2 and 5
        plt.plot(degrees, MSE_train, label="Train data MSE")
        plt.plot(degrees, MSE_test, label="Test data MSE")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Mean Square Error")
        plt.title(f"Train and test MSE as a function of model complexity for lambda = {l}")
        plt.legend()
        plt.savefig(f"plots/Ridge_test_train_lambda_{l}.pdf")
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
        plt.title(f"Error, Bias and Variances as function of model complexity for lambda = {l}")
        plt.legend()
        plt.savefig(f"plots/Ridge_error_bias_variance_lambda_{l}.pdf")
        plt.show()
        plt.clf()

        # Plot train & test MSE for degree up to 20
        k_fold_num = 5
        degreerange = 10
        degrees = range(1, degreerange + 1)
        MSE_train = []
        MSE_test = []

        for deg in degrees:
            X = rt.create_X_polynomial(x, y, deg)
            MSECV_train, MSECV_test = rt.CV_ridge(k_fold_num, X, z, l)
            MSE_train.append(np.mean(MSECV_train))
            MSE_test.append(np.mean(MSECV_test))

        # Plotting MSE and R2 for all polynomial orders between 2 and 5
        plt.plot(degrees, MSE_train, label="Train data MSE")
        plt.plot(degrees, MSE_test, label="Test data MSE")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Mean Square Error")
        plt.title(f"Train and test MSE as a function of model complexity - CV lambda {l}")
        plt.legend()
        plt.savefig(f"plots/Ridge_Cross_Validation_MSE_lambda_{l}.pdf")
        plt.show()
        plt.clf()
        print("Generated train v test MSE plot")


def problem_f():
    print("Problem f)")

def problem_g():
    print("Problem g)")

def main():
    problem_f()

if __name__ == "__main__":
    main()
# %%
