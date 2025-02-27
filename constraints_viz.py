# %%
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Dict, List
import mlflow
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from deepod.models.tabular import DeepSVDD, REPEN, RDP, RCA, GOAD, NeuTraL, ICL, SLAD
from tqdm import tqdm
from utils import downsample, dirac_mask

from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.datasets import dataset_factory

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# dataset = "ctu_13_neris"
dataset_name = "malware"
dataset = dataset_factory.get_dataset(dataset_name)
constraints_checker = ConstraintChecker(dataset.get_constraints(), tolerance=1e-3)

X, y = dataset.get_x_y()
X_down, y_down = downsample(X, y, frac=0.1)
X_train, X_test, y_train, y_test = train_test_split(
  X_down, y_down, test_size=0.2, random_state=42
)


# %%

def generate_hypercube():
  """Generate synthetic data in a hypercube
  Returns:
    pd.DataFrame: Synthetic data in a hypercube
  """
  # check if dtypes are either float or int
  min_vals = X_train.min()
  max_vals = X_train.max()
  dtypes = X_train.dtypes

  num_samples = 10**4
  synthetic_data = {}

  for column in X_train.columns:
    if dtypes[column] == "int64":
      # Generate uniform integers
      synthetic_data[column] = np.random.randint(
        low=min_vals[column],
        high=max_vals[column] + 1,  # +1 because upper bound is exclusive
        size=num_samples,
      )
    elif dtypes[column] == "float64":
      # Generate uniform floats
      synthetic_data[column] = np.random.uniform(
        low=min_vals[column], high=max_vals[column], size=num_samples
      )
    else:
      raise ValueError(f"Unsupported data type for column {column}")


# %%
# Create DataFrame from synthetic data
synthetic_data = generate_hypercube()
df_cube = pd.DataFrame(synthetic_data)

# Standardize the data for PCA/UMAP
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cube)

# Apply UMAP to reduce to 2 dimensions
# umap_model = UMAP(n_components=2, random_state=42)
umap_model = PCA(n_components=2, random_state=42)
umap_embedding = umap_model.fit_transform(scaled_data)

# Correctly check constraints only on the synthetic data
validity = constraints_checker.check_constraints(df_cube.to_numpy(), df_cube.to_numpy())

# Convert the validity result to a boolean mask (True: valid, False: invalid)
valid_mask = validity.astype(bool)

# Label the data points as "Valid" or "Invalid"
labels = np.array(["Invalid"] * len(df_cube))
labels[valid_mask] = "Valid"

# Create a color map for the labels
color_map = {"Valid": "green", "Invalid": "red"}
colors = [color_map[label] for label in labels]

# Plot the UMAP embedding
plt.figure(figsize=(10, 7))
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=colors, alpha=0.6, s=0.5)
plt.title("UMAP Projection of Synthetic Data")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

# Add legend
handles = [
  plt.Line2D(
    [0],
    [0],
    marker="o",
    color="w",
    label=label,
    markerfacecolor=color_map[label],
    markersize=10,
  )
  for label in color_map
]
plt.legend(handles=handles)
plt.show()
