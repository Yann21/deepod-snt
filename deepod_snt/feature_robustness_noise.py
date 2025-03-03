#%%
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Dict, List
import mlflow
from utils import downsample, dirac_mask

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from deepod.models.tabular import DeepSVDD, REPEN, RDP, RCA, GOAD, NeuTraL, ICL, SLAD
from tqdm import tqdm
from utils import downsample, dirac_mask
import os
from dotenv import load_dotenv

from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.datasets import dataset_factory

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# dataset_name = "ctu_13_neris"
dataset_name = "lcld_v2_iid"
dataset = dataset_factory.get_dataset(dataset_name)
constraints_checker = ConstraintChecker(dataset.get_constraints(), tolerance=1e-3)

X, y = dataset.get_x_y()
X_down, y_down = downsample(X, y, frac=0.1)
X_train, X_test, y_train, y_test = train_test_split(
  X_down, y_down, test_size=0.2, random_state=42
)


def generate_noisy_attacks(X, sigma_levels):
  n_columns = X.shape[1]
  mean = np.zeros(n_columns)
  data = np.zeros((len(sigma_levels), n_columns))

  for j, sigma in enumerate(tqdm(sigma_levels, desc="Processing noise levels")):
    cov = sigma * np.std(X.values, axis=0) * np.eye(n_columns)
    for i in range(n_columns):
      noise = np.random.multivariate_normal(mean, cov, X.shape[0])
      X_noisy = X + noise * dirac_mask(n_columns, i)
      is_valid = constraints_checker.check_constraints(X_noisy.values, X_noisy.values)
      invalid_percentage = 100 * np.sum(~is_valid) / X.shape[0]
      data[j, i] = invalid_percentage

  return pd.DataFrame(data, index=sigma_levels, columns=X.columns)


sigma_levels = np.logspace(-5, 5, num=11, base=10)

# Generate the 2D array of invalid percentages
invalid_percentages = generate_noisy_attacks(X_train, sigma_levels)

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
  invalid_percentages,
  annot=True,
  cmap="viridis",
  cbar_kws={"label": "Invalid Percentage (%)"},
)
plt.title("Invalid Percentage by Feature and Noise Level (Std)")
plt.xlabel("Feature")
plt.ylabel("Noise Level (Std)")
plt.show()

#%%
x = np.abs(invalid_percentages - 50)
y = x.idxmin(axis=0)
z = np.log10(y)

plt.figure(figsize=(10, 20))
plt.bar(x.columns, z)
plt.title("Feature Robustness to Noise")
plt.xlabel("Feature")
plt.ylabel("Log10(Noise Level)")
plt.xticks(rotation=90)
plt.xticks(fontsize=6)
plt.plot()

#%%
ser = z.value_counts().sort_index(ascending=False)
df = pd.DataFrame(ser).reset_index()
df.columns = ["Log Noise", "Feature Count"]
df.to_clipboard()

#%%
dataset.data