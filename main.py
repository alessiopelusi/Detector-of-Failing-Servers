import numpy as np
from utils import *

#Calculates mean and variance of all features in the dataset
def estimate_gaussian(X): 
    """
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape
    mu = 1/m * np.sum(X, axis = 0)
    var = 1/m * np.sum((X - mu)**2, axis = 0)
    
    return mu, var

# Finds the best threshold to use for selecting outliers based on the results from a validation set (p_val) and the ground truth (y_val)
def select_threshold(y_val, p_val): 
    """
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        
        predictions = p_val < epsilon
        tp = np.sum((predictions == 1) & (y_val==1))
        fp = np.sum((predictions == 0) & (y_val==1))
        fn = np.sum((predictions == 1) & (y_val==0))
        precision = tp / (tp + fp)
        recall =  tp / (tp + fn)
        F1 = (2 * precision * recall) / (precision + recall)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1


# Load the dataset
X_train, X_val, y_val = load_data()

# Estimate mean and variance of each feature
mu, var = estimate_gaussian(X_train)              

print("Mean of each feature:", mu)
print("Variance of each feature:", var)

# Returns the density of the multivariate normal
# at each data point (row) of X_train
p = multivariate_gaussian(X_train, mu, var)

p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)

# Find the outliers in the training set 
outliers = p < epsilon
print('# Anomalies found: %d'% sum(outliers))