import numpy as np

def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def generate_data_Franke(N, sigma2, seed):
    """Generates N uniformly distributed points on [0,1] x [0,1]
    applies the Franke Function to the points and adds noise

    Parameters
    ----------
    N: int
        Number of points
    sigma2: float
        Variance of error term 
    seed: int
        seed

    Returns
    -------
    x: np.ndarray
        N uniformly distributed x-coordinates
    y: np.ndarray
        N uniformly distributed y-coordinates
    z: np.ndarray
        Franke Function applied to x and y plus noise
    z_true:
        Franke Function applied to x and y, without noise
    """
    np.random.seed(seed)

    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    true_z = franke_function(x, y)
    epsilon = np.random.normal(0, sigma2, (N)) # generates n samples epsilon ~ N(0, sigma2)
    z = true_z + epsilon

    return x, y, z, true_z

