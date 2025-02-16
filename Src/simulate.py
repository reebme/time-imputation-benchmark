import numpy as np
from numpy.polynomial import Polynomial as P
from itertools import combinations

def generate_stationary_roots(p):
    """
    Generate p roots for the characteristic polynomial of an AR(p) process.
    
    The characteristic polynomial follows the Z-transform representation:
        z^p - phi_1*z^(p-1) - ... - phi_(p-1)*z - phi_p = 0

    Conditions for stationarity:
    - All roots must be inside the unit circle (|z| < 1).
    - Complex roots appear in conjugate pairs.
    - If p is odd, at least one root must be real.

    Parameters:
    p (int): Order of the AR process.

    Returns:
    ndarray: Array of stationary roots.
    """
    rng = np.random.default_rng()
    roots = []
    
    num_complex_pairs = p // 2  # Number of complex conjugate pairs
    num_real = p % 2  # If p is odd, we need one real root

    # Small lower bound to avoid trivial values
    lower_bound = 1e-8

    # Generate complex roots in conjugate pairs
    for _ in range(num_complex_pairs):
        r = rng.uniform(lower_bound, 1)  # Ensure |z| < 1
        theta = rng.uniform(0, np.pi)  # Random angle (0 to pi ensures unique conjugate)
        complex_root = r * np.exp(1j * theta)
        roots.append(complex_root)
        roots.append(np.conj(complex_root))  # Add conjugate pair

    # If p is odd, add a real root
    if num_real == 1:
        real_root = rng.uniform(lower_bound, 1)
        roots.append(real_root)

    return np.array(roots)

def compute_poly_coeffs_old(roots):
    return np.poly(roots)[1:]


def compute_poly_coeffs_new(roots):
    p = P.fromroots(roots)
    poly_coeffs = p.coef

    return np.flip(poly_coeffs[:-1])

def vieta_brute_force(roots):
    """
    Demonstrates Vieta's formulas by computing the coefficients
    of the monic polynomial with the given roots.
    
    For a polynomial with p roots each coefficient (for k = 1, …, p)
    is the sum of the products of all k-element combinations of the roots:

    a_k = sum (z_{i1} * z_{i2} * ... * z_{ik}) 
    over all k-element subsets {i1, i2, …, ik} of {1, 2, …, p}.

    By Vieta’s formulas, the coefficient of x^(p-k) is (-1)^k * a_k.

    NOTE: This implementation explicitly generates all k-element combinations,
    leading to EXPONENTIAL COMPLEXITY.
    It is for educational purposes only and should not be used for large inputs.
    """
    p = len(roots)
    coefficients = []  # will hold the coefficients for x^(p-1), x^(p-2), ..., x^0

    # Loop over k = 1, 2, ..., p (number of roots to multiply)
    for k in range(1, p+1):
        a_k = 0
        # Iterate over all combinations of roots taken k at a time.
        for combo in combinations(roots, k):
            product = 1
            for z in combo:
                product *= z
            a_k += product
        # According to Vieta's formulas, the coefficient for x^(p-k) is (-1)^k * phi_k.
        coefficient = (-1)**(k) * a_k
        coefficients.append(coefficient)
    return coefficients

def calculate_vietas_coeffs(roots):
    """
    Computes the coefficients of a monic polynomial given its roots
    using Vieta's formulas. For a polynomial with p roots:

    (x - r1) * (x - r2) * ... * (x - rp) 
    = x^p - phi_1*x^(p-1) + phi_2*x^(p-2) - ... + (-1)^p * phi_p,

    each phi_k (for k = 1, …, p) is the sum of the products
    of all k-element combinations of the roots:

    phi_k = sum (r_{i1} * r_{i2} * ... * r_{ik}) 
    over all k-element subsets {i1, i2, …, ik} of {1, 2, …, p}.

    By Vieta’s formulas, the coefficient of x^(p-k) is (-1)^k * phi_k.

    This function computes the phi_k values using dynamic programming,
    avoiding the exponential complexity of a brute-force approach.
    """
    p = len(roots)
    # Initialize the list: phi[0] = 1, phi[1...n] = 0
    phi = [1] + [0] * p

    # Update phi for each root
    for r in roots:
        for j in range(p, 0, -1):
            phi[j] -= r * phi[j-1]

    # phi[k] is (-1)**k * (sum of products of k roots)
    # Print the values for k = 1 to n (matching the original code's output)
    return phi[1:]

def generate_ar_process(ar_params, T, burnin_factor = 10, sigma = 1.0, random_seed = None):
    """
    Generate a synthetic AR(p) time series of length T using the given coefficients.
    
    Parameters
    ----------
    ar_params : array-like
        AR coefficients, e.g. [phi_1, phi_2, ..., phi_p] for AR(p).
        The model is X_t = sum_{i=1..p} phi_i * X_{t-i} + e_t
    T : int
        Number of output samples desired for the AR(p) process.
    burnin_factor : int, optional
        How many multiples of p to use as burn-in. Defaults to 10.
    sigma : float, optional
        Standard deviation of the white noise. Defaults to 1.0.
    random_seed : int, optional
        Seed for the random number generator (for reproducibility).

    Returns
    -------
    x : ndarray
        A 1D numpy array of length T containing the generated AR(p) process.
    noise: ndarray
        Noise used to generate the synthetic process.
    """

    rng = np.random.default_rng(random_seed)
    
    p = len(ar_params)               # order p
    burnin = burnin_factor * p       # number of burn-in samples
    
    # Total samples to generate (including burn-in)
    total_length = T + burnin
    
    # Allocate array for the AR process
    x = np.zeros(total_length)
    
    # Generate white noise
    noise = rng.normal(loc = 0.0, scale = sigma, size = total_length)
   
    #TODO benchmark against pandas shift and calculate 
    # AR(p): X[t] = phi_1*X[t-1] + ... + phi_p*X[t-p] + noise[t]
    for t in range(p, total_length):
        x[t] = np.flip(x[t-p:t]) @ ar_params + noise[t]
    
    # Discard the burn-in samples
    return x[burnin:], noise[burnin:]
