# %%
import pandas as pd
import gdown
from pathlib import Path
import numpy as np

here = Path(__file__).parent


def download_all_data() -> None:
  folder = "https://drive.google.com/drive/folders/12xtPwtW5sZUxe_hxWKG-U_zdNwuYbF8q?usp=drive_link"
  gdown.download_folder(folder)


def load_data(dataset, logger):
  """Load dataset from file."""
  logger.info(f"Loading data for dataset: {dataset}")
  df = pd.read_csv(here / f"data/constrained/{dataset}/data.csv")
  return df.drop(columns=["target"]), df["target"]


def downsample(X, y, frac=0.1):
  """Downsample dataset."""
  sample_size = int(frac * len(X))
  indices = np.random.choice(X.index, sample_size, replace=False)

  X_down = X.iloc[indices]
  y_down = y[indices]
  return X_down, y_down


def dirac_mask(size, i):
  """
  >>> dirac_mask(5, 3)
  array([0., 0., 1., 0., 0.])
  """
  # Fun trick where dirac in i is the ith row of the identity matrix.
  return np.eye(size)[i]
