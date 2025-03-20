# %%
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import downsample, dirac_mask
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.datasets import dataset_factory
from typing import Union
from nptyping import NDArray, Shape

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# dataset_name = "ctu_13_neris"
dataset_name = "lcld_v2_iid"
dataset = dataset_factory.get_dataset(dataset_name)
constraints_checker = ConstraintChecker(dataset.get_constraints(), tolerance=1e-3)


def generate_noisy_attacks(
  X: pd.DataFrame, sigma_levels: np.ndarray, pairwise: bool
) -> np.ndarray:
  """
  Generate noisy attacks feature-wise or pairwise. Return the percentage of
  data points that become invalid after adding the noise.

  In a pairwise setting, we achieve perfect correlation in noise in the
  following way:
  $$
  \Sigma =
  \begin{bmatrix}
  \sigma_1^2 & \sigma_1 \sigma_2 \\
  \sigma_1 \sigma_2 & \sigma_2^2
  \end{bmatrix}
  $$
  """
  n_columns = X.shape[1]
  mean = np.zeros(n_columns)
  stds = np.std(X.values, axis=0)

  if pairwise:
    data = np.zeros((len(sigma_levels), n_columns, n_columns))
  else:
    data = np.zeros((len(sigma_levels), n_columns))

  for j, sigma in enumerate(
    tqdm(sigma_levels, desc="Processing noise levels (10-20min)")
  ):
    cov = sigma * np.eye(n_columns) * stds  # Start with diagonal covariance

    for i in range(n_columns):
      if pairwise:
        for k in range(i, n_columns):  # Compute only upper triangular part
          cov_pairwise = cov.copy()
          if i != k:
            # Have perfect correlation between i and k
            cov_pairwise[i, k] = cov_pairwise[k, i] = stds[i] * stds[k]

          noise = np.random.multivariate_normal(mean, cov_pairwise, X.shape[0])
          # Add noise on both feature i and k
          X_noisy = X + noise * (dirac_mask(n_columns, i) + dirac_mask(n_columns, k))
          is_valid = constraints_checker.check_constraints(
            X_noisy.values, X_noisy.values
          )
          invalid_percentage = 100 * np.sum(~is_valid) / X.shape[0]
          data[j, i, k] = data[j, k, i] = invalid_percentage  # Symmetric matrix

      else:
        noise = np.random.multivariate_normal(mean, cov, X.shape[0])
        X_noisy = X + noise * dirac_mask(n_columns, i)
        is_valid = constraints_checker.check_constraints(X_noisy.values, X_noisy.values)
        invalid_percentage = 100 * np.sum(~is_valid) / X.shape[0]
        data[j, i] = invalid_percentage

  return data


X, y = dataset.get_x_y()
X_down, y_down = downsample(X, y, frac=0.01)
X_train, X_test, y_train, y_test = train_test_split(
  X_down, y_down, test_size=0.2, random_state=42
)
sigma_levels = np.logspace(-5, 5, num=11, base=10)

invalidity_percentages = generate_noisy_attacks(X_train, sigma_levels, pairwise=True)
distance_to_50 = np.abs(invalidity_percentages - 50)
std_closest_to_50 = np.argmin(distance_to_50, axis=0)
# std_closest_to_50 = closest_to_50.idxmin(axis=0)
# log_std_closest_to_50 = np.log10(std_closest_to_50)
log_std_closest_to_50 = np.log10(sigma_levels[std_closest_to_50])
log_std_closest_to_50


# %% Pairwise visualization
plt.figure(figsize=(18, 15))
sns.heatmap(
  log_std_closest_to_50, cmap="YlGnBu", xticklabels=X.columns, yticklabels=X.columns
)


# %% Feature-wise visualization
plt.figure(figsize=(15, 15))
plt.bar(X.columns, log_std_closest_to_50)
plt.title("Feature Robustness to Noise")
plt.xlabel("Feature")
plt.ylabel("Log10(Noise Level)")
plt.xticks(rotation=90)
plt.xticks(fontsize=6)
plt.plot()


# %% Count the number of features that are most robust to noise
ser = log_std_closest_to_50.value_counts().sort_index(ascending=False)
df = pd.DataFrame(ser).reset_index()
df.columns = ["Log Noise", "Feature Count"]
df
