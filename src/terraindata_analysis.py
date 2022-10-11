# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler

import regression_tools as rt
from imageio import imread

def plot_terrain():
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


def OLS_regression(x, y, z):
    """ Does OLS """

    # set parameters
    deg = 5 # degree of polynomial fit

    print(f"Linear regression with ols, polynomial of degree={deg}:")
    print()

    # Create X design matrix with coordinates x and y 
    X = rt.create_X_polynomial(x, y, deg)
    #Scale data
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    z_s = scaler.fit_transform(z.reshape(-1, 1)).flatten()

    # Test-train-split both X-design matrix and z-data
    X_train_s, X_test_s, z_train_s, z_test_s = train_test_split(X_s, z_s, test_size=0.25)

    # Estimate beta with OLS linear regression
    ols = rt.LinearRegression()
    beta = ols(X_train_s, z_train_s)

    # Use model to predict and descale prediction
    ztilde_train_s = X_train_s @ beta
    ztilde_test_s = X_test_s @ beta 

    print("R2 - training")
    print(rt.R2(z_train_s, ztilde_train_s))
    print("MSE - training")
    print(rt.MSE(z_train_s, ztilde_train_s))
    print()
    print("R2 - testing")
    print(rt.R2(z_test_s, ztilde_test_s))
    print("MSE - testing")
    print(rt.MSE(z_test_s, ztilde_test_s))
    print()


def OLS_regression_CV(x, y, z):
    """Cross validation on data for a range of degrees"""
    """Takes 20sek with current parameters"""

    # Set parameters 
    k_fold_num = 5
    degreerange = 40
    degrees = range(5, degreerange + 1)

    #lists to obtain MSE-values for different degrees
    MSE_train = []
    MSE_test = []

    print(f"Model = linreg, resampling = CV, k-fold={k_fold_num}")
    print("Degrees, Mean(MSE-Train), Mean(MSE-test)")

    #progress = 0
    #print(f'progress = {progress}')
    for deg in degrees:
        X = rt.create_X_polynomial(x, y, deg)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        z_s = scaler.fit_transform(z.reshape(-1, 1)).flatten()

        #progress = deg/degreerange * 100
        #print(f"Progress: {round(progress, 2)}%")
        MSECV_train, MSECV_test = rt.cross_validation(X_s, z_s, k_fold_num, rt.ols_regression)
        MSE_train.append(np.mean(MSECV_train))
        MSE_test.append(np.mean(MSECV_test))
        print(f'{deg}, {np.mean(MSECV_train)}, {np.mean(MSECV_test)}')

    # Plotting MSE for all polynomial orders
    plt.plot(degrees, MSE_train, "--o" ,label="Train data MSE")
    plt.plot(degrees, MSE_test, "--o" ,label="Test data MSE")
    plt.xlabel("Polynomial degree", fontsize=14)
    plt.ylabel("Mean Square Error", fontsize=14)
    plt.title(f"MSE as a function of polynomial degree \n w/Cross validation - [n={n}, kfold={k_fold_num}]", fontsize=16)
    plt.xticks(ticks=degrees,fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.legend(fontsize=12)
    #plt.savefig(f"../plots/Oppgave_g/CV_deg={degreerange}_N={n}_Kfold={k_fold_num}.pdf")
    plt.show()
    plt.clf()
    

def ridge_regression_CV(x, y, z):
    """Does ridge-reggresion for different lambda
        Takes 3min with current parameteres
    """

    # Set parameters
    lambdas = np.array([0.001, 0.01, 0.3, 0.5, 0.8, 1, 10_000])
    k_fold_num = 5
    degreerange = 10
    degrees = range(4, degreerange + 1)


    print(f"Model = lasso, resampling = CV, kfold = {k_fold_num}'")
    print("Lambda , Degrees, Mean(MSE-Train), Mean(MSE-test)")

    for l in lambdas:
        # Set parameters 
        MSE_train = []
        MSE_test = []

        for deg in degrees:
            print(deg)
            X = rt.create_X_polynomial(x, y, deg)
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            z_s = scaler.fit_transform(z.reshape(-1, 1)).flatten()

            ridge = rt.LinearRegression('ridge', l)
            MSECV_train, MSECV_test = rt.cross_validation(X_s, z_s, k_fold_num, ridge)
            MSE_train.append(np.mean(MSECV_train))
            MSE_test.append(np.mean(MSECV_test))
            print(f'{l}, {deg}, {np.mean(MSECV_train)}, {np.mean(MSECV_test)}')


        # Plotting MSE and R2 for all polynomial orders between 2 and 5
        plt.plot(degrees, MSE_train, '--o', label="Train data MSE")
        plt.plot(degrees, MSE_test, '--o' ,label="Test data MSE")
        plt.xlabel("Polynomial degree", fontsize=14)
        plt.ylabel("Mean Square Error", fontsize=14)
        plt.title(f"MSE as a function of polynomial degree \n Cross validation w/Ridge - [$\lambda$={l}, n={n}, kfold={k_fold_num}]", fontsize=16)
        plt.legend(fontsize=14)
        plt.xticks(ticks=degrees,fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        #plt.savefig(f"../plots/Oppgave_g/E_Ridge_lambda={l}_degrange={degreerange}_N={n}_Kfold={k_fold_num}_CV_MSE.pdf")
        plt.show()
        plt.clf()


def ridge_over_lambda(x, y, z):
    # Set parameters:
    k_fold_num = 5
    degrees = [50]

    print(f"Model = lasso, resampling = CV, k-fold={k_fold_num}")
    print("Now inverse, lambda on x-axis for certian degrees")
    print("Lambda , Degree, Mean(MSE-Train), Mean(MSE-test)")
    
    for deg in degrees:
        X = rt.create_X_polynomial(x, y, deg)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        z_s = scaler.fit_transform(z.reshape(-1, 1)).flatten()


        lambdalogspace = np.logspace(-3, 10)
        MSE_train = []
        MSE_test = []

        for l in lambdalogspace:
            model = rt.LinearRegression('ridge', l)
            MSECV_train, MSECV_test = rt.cross_validation(X_s, z_s, k_fold_num, model)
            MSE_train.append(np.mean(MSECV_train))
            MSE_test.append(np.mean(MSECV_test))
            print(f'{l}, {deg}, {np.mean(MSECV_train)}, {np.mean(MSECV_test)}')

        # Plotting MSE and R2 for all polynomial orders between 2 and 5
        plt.plot(np.log10(lambdalogspace), MSE_train, label="Train data MSE")
        plt.plot(np.log10(lambdalogspace), MSE_test, label="Test data MSE")
        plt.xlabel(r"Log($\lambda$)", fontsize=12)
        plt.ylabel("Mean Square Error", fontsize=12)
        plt.title(f"MSE as a function of log(lambda) \n Cross validation w/Ridge-[poly_deg={deg},n={n}, kfold={k_fold_num}]", fontsize=15)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        #plt.savefig(f"../plots/Oppgave_g/ridge/deg={deg}_N={n}_Kfold={k_fold_num}.pdf")
        plt.show()


def lasso_regression_CV(x, y, z):
    """Lasso-reggresion for different lambda """
    """Takes approx 8min with current parameters"""
    
    # Set parameters 
    lambdas = np.array([1e-9, 1e-8, 1e-7, 1e-4])
    k_fold_num = 5
    degreerange = 8
    degrees = range(4, degreerange + 1)

    print(f"Model = lasso, resampling = CV, k-fold={k_fold_num}")
    print("Lambda , Degrees, Mean(MSE-Train), Mean(MSE-test)")

    for l in lambdas:
        MSE_train = []
        MSE_test = []

        for deg in degrees:
            X = rt.create_X_polynomial(x, y, deg)
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            z_s = scaler.fit_transform(z.reshape(-1, 1)).flatten()

            lasso = rt.LinearRegression('lasso', l)
            MSECV_train, MSECV_test = rt.cross_validation(X_s, z_s, k_fold_num, lasso)
            MSE_train.append(np.mean(MSECV_train))
            MSE_test.append(np.mean(MSECV_test))
            #print(f'{l}, {deg}, {np.mean(MSECV_test)} , {np.mean(MSECV_train)}')

        # Plotting MSE and R2 for all polynomial orders between 2 and 5
        plt.plot(degrees, MSE_train,'--o', label="Train data MSE")
        plt.plot(degrees, MSE_test,'--o' ,label="Test data MSE")
        plt.xlabel("Polynomial degree", fontsize=14)
        plt.ylabel("Mean Square Error", fontsize=14)
        plt.title(f"MSE as a function of polynomial degree w/CV\n Lasso-[lambda={l}, N={n}, k-fold={k_fold_num}]", fontsize=15)
        plt.legend(fontsize=12)
        plt.xticks(ticks=degrees,fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        #plt.savefig(f"../plots/Oppgave_g/F_Lasso_lambda={l}_degrange={degreerange}_N={n}_Kfold={k_fold_num}_CV_MSE.pdf")
        plt.show()
        plt.clf()


def lasso_over_lambda(x, y, z):
    # Set parameters 

    k_fold_num = 5
    degrees = [8, 10, 12, 16]

    print(f"Model = lasso, resampling = CV, k-fold={k_fold_num}")
    print("Now inverse, lambda on x-axis for certian degrees")
    print("Lambda , Degree, Mean(MSE-Train), Mean(MSE-test)")
    
    for deg in degrees:
        X = rt.create_X_polynomial(x, y, deg)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        z_s = scaler.fit_transform(z.reshape(-1, 1)).flatten()


        lambdalogspace = np.logspace(-5, 2)
        MSE_train = []
        MSE_test = []

        for l in lambdalogspace:
            model = rt.LinearRegression('lasso', l)
            MSECV_train, MSECV_test = rt.cross_validation(X_s, z_s, k_fold_num, model)
            MSE_train.append(np.mean(MSECV_train))
            MSE_test.append(np.mean(MSECV_test))
            print(f'{l}, {deg}, {np.mean(MSECV_train)}, {np.mean(MSECV_test)}')

        # Plotting MSE and R2 for all polynomial orders between 2 and 5
        plt.plot(np.log10(lambdalogspace), MSE_train,label="Train data MSE")
        plt.plot(np.log10(lambdalogspace), MSE_test,  label="Test data MSE")
        plt.xlabel(r"Log($\lambda$)", fontsize=14)
        plt.ylabel("Mean Square Error", fontsize=14)
        plt.title(f"MSE as a function of log(\lambda) \n Cross validation w/Lasso-[poly_deg={deg},n={n},kfold={k_fold_num}]", fontsize=15)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        #plt.savefig(f"../plots/Oppgave_g/lasso/lasso-deg={deg}_N={n}_Kfold={k_fold_num}_CV_MSE.pdf")
        plt.show()
        plt.clf()


def main(tif_file):
    #We constrain our number of data by randomly picking out 1000 points
    global n
    n = 1000
    print(f'Number of randomly selected points = {n}')
    np.random.seed(2018)
    tif = imread(tif_file)
    x = []
    y = []
    z = []
    for N in range(n):
        x_val = np.random.randint(0, len(tif[0]))
        y_val = np.random.randint(0, len(tif))
        x.append(x_val)
        y.append(y_val)
        z.append(tif[y_val][x_val])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    lasso_regression_CV(x, y, z)

if __name__ == "__main__":
    main("SRTM_Saarland.tif")
# %%