import numpy as np
import scipy.stats as stats

def normiz(x):
    """
    Normalize array x between 0 and 1
    
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def G(x, mu, sigma):
    """
    Gaussian with standard deviation sigma, mean mu, and no multiplicative constant
    
    inputs
    -------------------
    x: N x 1 array
        wavelength range  
    mu: float
        central wavelength of the feature
    sigma: float
        standard deviation of the line
        
    outputs
    --------------------
    N x 1 array
        evaluation of Gaussian function at every wavelength x
    
    """
    
    return np.exp(-(x - mu)**2/(2*sigma**2))


def gen_spec(linelist, x, K = 9, noise = 0.1, continuum = 1.0):
    """
    Generate a spectrum of K random lines from a known line list, each with a Gaussian line profile
    
    inputs
    ------------------
    linelist: array
        known mean wavelength of absorption lines of interest
    x: N x 1 array 
        wavelength range   
    K: int
        number of features (lines) to include 
    noise: float
        amount of Gaussian noise to add to the "true" spectrum
    continuum: float
        level of continuum emission (amount by which to shift the spectrum)
    
    
    outputs
    -------------------
    features: N x K array
        array containing each individual "true" feature without noise
    spectrum: N x 1 array
        value of the reflectance at each wavelength (i.e., the sum of all of the individual features)
    
    """
    
    
    which = np.sort(np.random.choice(np.arange(0,9,1), size = K, replace = False))   # pick the lines
    lines = linelist[which]                                                          # ^
    
    phis = stats.beta.rvs(2, 3, size = K)                                            # get a depth for each line
    #print(phis)
    means = lines + stats.norm.rvs(0,1e-3, size = K)                                 # let each line wiggle slightly around the true mean
    
    sigmas = np.random.gamma(2, 0.02, size = K)                                 # get a width for each line
    
    # now make the line profiles
    f = []
    for i in range(K):
        f.append(-phis[i] * G(x, means[i], sigmas[i]))
    
    # write down the true features
    features = {'f':np.array(f) + continuum,
                'which':np.sort(which),
                'phi':phis,
                'mu':means,
                'sigma':sigmas}

    # add some noise
    fuzz = stats.norm.rvs(0, noise, size = x.size)
    
    # and now you have a spectrum
    spectrum = np.sum(f, axis = 0) + fuzz + continuum
    
    return features, np.clip(spectrum, a_min = 0, a_max = None)
