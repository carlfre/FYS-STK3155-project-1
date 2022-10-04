# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

import Testing as rt
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

def problem_b(tif_file):
    print("Problem g) part b):")
    tif = imread(tif_file)
    x = []
    y = []
    z = []
    for N in range(1_000_000):
        x_val = np.random.randint(0, len(tif[0]))
        y_val = np.random.randint(0, len(tif))
        x.append(x_val)
        y.append(y_val)
        z.append(tif[y_val][x_val])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    """ Code for all points in tif-file
    x = np.zeros([len(tif[0]), len(tif)])
    for i in range(len(x)):
        x[i] = np.array([i]*len(tif[0]))
    y = np.zeros([len(tif[0]), len(tif)])
    for i in range(len(y)):
        y[i] = np.array([i] * len(tif[0]))
    y = y.T
    z = tif.flatten()
    """

    # Create design matrix and train-test-split
    # degree of polynomial
    deg = 5
    X = rt.create_X_polynomial(x, y, deg)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    z_train_scaled = z_train - np.mean(z_train)
    z_test_scaled = z_test - np.mean(z_train)

    # Estimate beta
    beta = rt.linreg(X_train_scaled, z_train_scaled)

    # Evaluate MSE and R2
    ztilde_train = X_train_scaled @ beta + np.mean(z_train)
    ztilde_test = X_test_scaled @ beta + np.mean(z_train)

    """ Code for plotting all points
    ztilde = X_scaled @ beta
    ztilde_reshaped = np.reshape(ztilde, [len(tif[0]), len(tif)])
    plt.figure()
    plt.title(f"Terrain over Saarland with degree {deg}")
    plt.imshow(ztilde_reshaped, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"regression_all_points_degree={deg}.pdf")
    plt.show()
    """

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





def problem_d(tif_file):
    print("Problem d)")
    np.random.seed(569)

    tif = imread(tif_file)
    x = []
    y = []
    z = []
    for N in range(10_000):
        x_val = np.random.randint(0, len(tif[0]))
        y_val = np.random.randint(0, len(tif))
        x.append(x_val)
        y.append(y_val)
        z.append(tif[y_val][x_val])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Plot train & test MSE for degree up to 20
    k_fold_num = 5
    degreerange = 30
    degrees = range(2, degreerange + 1)
    MSE_train = []
    MSE_test = []

    progress = 0
    print("Progress: 0%")
    for deg in degrees:
        X = rt.create_X_polynomial(x, y, deg)
        scaler = StandardScaler()
        scaler.fit(X)
        scaler.transform(X)
        z = z - np.mean(z)
        progress = deg/degreerange * 100
        print(f"Progress: {round(progress, 2)}%")


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
    #plt.savefig("plots/Cross_Validation_MSE.pdf")
    plt.show()
    plt.clf()
    print("Generated train v test MSE plot")


def problem_e(tif_file):
    print("Problem e)")
    # Preform c-d whith ridge-reggresion for different lambda
    lambdas = np.array([0.00001, 0.001, 0.1, 10])

    np.random.seed(569)

    # Generating data
    tif = imread(tif_file)
    x = []
    y = []
    z = []
    for N in range(120_000):
        x_val = np.random.randint(0, len(tif[0]))
        y_val = np.random.randint(0, len(tif))
        x.append(x_val)
        y.append(y_val)
        z.append(tif[y_val][x_val])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    for l in lambdas:
        # Plot train & test MSE for degree up to 20
        degreerange = 10
        degrees = range(2, degreerange + 1)
        MSE_train = []
        MSE_test = []

        for deg in degrees:
            X = rt.create_X_polynomial(x, y, deg)
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            z_train = z_train - np.mean(z_train)
            z_test = z_test - np.mean(z_test)

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
        #plt.savefig(f"plots/Ridge_test_train_lambda_{l}.pdf")
        plt.show()
        plt.clf()
        print("Generated train v test MSE plot")



        # Plot train & test MSE for degree up to 20
        k_fold_num = 5
        degreerange = 10
        degrees = range(2, degreerange + 1)
        MSE_train = []
        MSE_test = []

        for deg in degrees:
            X = rt.create_X_polynomial(x, y, deg)
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            z = z - np.mean(z)

            MSECV_train, MSECV_test = rt.CV_ridgereg(k_fold_num, X, z, l)
            MSE_train.append(np.mean(MSECV_train))
            MSE_test.append(np.mean(MSECV_test))

        # Plotting MSE and R2 for all polynomial orders between 2 and 5
        plt.plot(degrees, MSE_train, label="Train data MSE")
        plt.plot(degrees, MSE_test, label="Test data MSE")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Mean Square Error")
        plt.title(f"Train and test MSE as a function of model complexity - CV lambda {l}")
        plt.legend()
        #plt.savefig(f"plots/Ridge_Cross_Validation_MSE_lambda_{l}.pdf")
        plt.show()
        plt.clf()
        print("Generated train v test MSE plot")



def problem_f(tif_file):
    print("Problem e)")
    # Preform c-d whith lasso-reggresion for different lambda
    degrees = np.array([2, 4, 6, 8])

    np.random.seed(569)

    # Generating data
    tif = imread(tif_file)
    x = []
    y = []
    z = []
    for N in range(10_000):
        x_val = np.random.randint(0, len(tif[0]))
        y_val = np.random.randint(0, len(tif))
        x.append(x_val)
        y.append(y_val)
        z.append(tif[y_val][x_val])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    for deg in degrees:
        # Plot train & test MSE for degree up to 20
        lambdas = np.logspace(-6, 1)
        MSE_train = []
        MSE_test = []

        X = rt.create_X_polynomial(x, y, deg)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        z_train = z_train - np.mean(z_train)
        z_test = z_test - np.mean(z_test)

        for l in lambdas:
            beta = rt.lassoreg(X_train, z_train, l)
            ztilde_train = (X_train @ beta) + np.mean(z_train)
            ztilde_test = (X_test @ beta) + np.mean(z_test)

            MSE_train.append(rt.MSE(z_train, ztilde_train))
            MSE_test.append(rt.MSE(z_test, ztilde_test))

        # Plotting MSE and R2 for all polynomial orders between 2 and 5
        plt.plot(np.log10(lambdas), MSE_train, label="Train data MSE")
        plt.plot(np.log10(lambdas), MSE_test, label="Test data MSE")
        plt.xlabel("Lambda")
        plt.ylabel("Mean Square Error")
        plt.title(f"Train and test MSE as a function of model complexity for degree = {deg}")
        plt.legend()
        # plt.savefig(f"plots/lasso_test_train_lambda_{l}.pdf")
        plt.show()
        plt.clf()
        print("Generated train v test MSE plot")

        # Plot train & test MSE for degree up to 20
        k_fold_num = 5
        MSE_train = []
        MSE_test = []
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        z = z - np.mean(z)

        for l in lambdas:
            MSECV_train, MSECV_test = rt.CV_lassoreg(k_fold_num, X, z, l)
            MSE_train.append(np.mean(MSECV_train))
            MSE_test.append(np.mean(MSECV_test))

        # Plotting MSE and R2 for all polynomial orders between 2 and 5
        plt.plot(np.log10(lambdas), MSE_train, label="Train data MSE")
        plt.plot(np.log10(lambdas), MSE_test, label="Test data MSE")
        plt.xlabel("Lambda")
        plt.ylabel("Mean Square Error")
        plt.title(f"Train and test MSE as a function of model complexity - CV degree {deg}")
        plt.legend()
        # plt.savefig(f"plots/lasso_Cross_Validation_MSE_lambda_{l}.pdf")
        plt.show()
        plt.clf()


def problem_g():
    """Test on real data"""
    # Load the terrain
    terrain1 = imread("SRTM_Saarland.tif")
    print(terrain1)
    # Show the terrain
    plt.figure()
    plt.title("Terrain over Saarland")
    plt.imshow(terrain1, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def main():
    problem_b(tif_file="SRTM_Saarland.tif")


if __name__ == "__main__":
    main()
# %%
