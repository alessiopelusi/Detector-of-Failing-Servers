# Detector-of-Failing-Servers

This project is designed for anomaly detection using a multivariate Gaussian distribution. It includes the following files:

1. `main.py`: The main script for estimating the Gaussian parameters, selecting a threshold, and identifying anomalies.

2. `utils.py`: A utility script containing functions for loading data and computing multivariate Gaussian probabilities.

## `main.py`

### estimate_gaussian(X)

This function calculates the mean and variance of all features in the dataset.

- Input:
  - `X` (ndarray): Data matrix of shape (m, n), where `m` is the number of data points and `n` is the number of features.

- Output:
  - `mu` (ndarray): Mean of all features, a vector of shape (n,).
  - `var` (ndarray): Variance of all features, a vector of shape (n,).

### select_threshold(y_val, p_val)

This function finds the best threshold to use for selecting outliers based on the results from a validation set (`p_val`) and the ground truth (`y_val`).

- Input:
  - `y_val` (ndarray): Ground truth on the validation set.
  - `p_val` (ndarray): Results on the validation set.

- Output:
  - `epsilon` (float): Threshold chosen.
  - `F1` (float): F1 score by choosing `epsilon` as the threshold.

### Other functionality

The script loads the dataset, estimates the mean and variance of each feature, calculates the density of the multivariate normal distribution, and identifies outliers in the training set.

## `utils.py`

### load_data()

This function loads the training data (`X`), validation data (`X_val`), and validation labels (`y_val`) from the data files.

- Output:
  - `X` (ndarray): Training data.
  - `X_val` (ndarray): Validation data.
  - `y_val` (ndarray): Validation labels.

### multivariate_gaussian(X, mu, var)

This function computes the probability density function of the examples in `X` under the multivariate Gaussian distribution with parameters `mu` and `var`. It can handle both a full covariance matrix or a diagonal covariance matrix.

- Input:
  - `X` (ndarray): Data matrix of shape (m, n), where `m` is the number of data points and `n` is the number of features.
  - `mu` (ndarray): Mean of the distribution, a vector of shape (n,).
  - `var` (ndarray): Covariance matrix (full or diagonal).

- Output:
  - `p` (ndarray): Probability densities for each data point in `X`.

## Usage

To use the anomaly detection system, run `main.py`, which will load the data, estimate Gaussian parameters, and identify anomalies in the training set.

For further details on the implementation, please refer to the code and function comments in both `main.py` and `utils.py`.

Enjoy using this anomaly detection system!
