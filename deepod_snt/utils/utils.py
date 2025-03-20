# %%
import pandas as pd
from pathlib import Path
import numpy as np
import os
import mlflow
import logging

here = Path(__file__).parent


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


def mlflow_log(experiment_name, params, metrics):
  username = os.environ["MLFLOW_USERNAME"]
  password = os.environ["MLFLOW_PASSWORD"]
  address = os.environ["MLFLOW_ADDRESS"]
  mlflow.set_tracking_uri(f"http://{username}:{password}@{address}")

  mlflow.set_experiment(experiment_name)
  with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)


def create_logger():
  logging.basicConfig()
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  return logger


def confirm_base_data_is_valid(constraints_checker, X_train, logger):
  is_valid = constraints_checker.check_constraints(X_train.values, X_train.values)
  X_train_valid, X_train_invalid = X_train[is_valid], X_train[~is_valid]
  native_invalid_prop = X_train_invalid.shape[0] / X_train.shape[0] * 100

  if native_invalid_prop > 5:
    logger.warning(
      f"Native invalid proportion: {native_invalid_prop:.2f}%. Goes against assumption that data is valid."
    )
