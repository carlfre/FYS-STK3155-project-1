#%%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

import regression_tools as rt



def problem_b():
    print("Problem b)")

    #TODO fix seed

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
    plt.plot(degrees, R2_arr)
    plt.xlabel("Polynomial order")
    plt.ylabel("R2-score")
    plt.title("Change in R2 with increasing polynomial order")
    plt.savefig("plots/R2.pdf")
    plt.show()
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
    #TODO fix seed

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
    print("Generated train v test MSE plot")


    #TODO


def problem_d():
    print("Problem d)")

def problem_e():
    print("Problem e)")

def problem_f():
    print("Problem f)")

def problem_g():
    print("Problem g)")

def main():
    problem_c()

if __name__ == "__main__":
    main()