# %% Import data
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Dict, List

from deepod.models.tabular import DeepSVDD, REPEN, RDP, RCA, GOAD, NeuTraL, ICL, SLAD
from tqdm import tqdm
from utils import downsample, dirac_mask, mlflow_log
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
#%%
X_train.shape

#%%
def generate_noisy_attacks_by_features(
  X: pd.DataFrame, sigma: float = 1
) -> Dict[str, List[pd.DataFrame]]:
  n_columns = X.shape[1]

  mean = np.zeros(n_columns)
  cov = sigma * np.std(X.values, axis=0) * np.eye(n_columns)

  data: Dict[str, List[pd.DataFrame]] = {"valid": [], "invalid": []}
  for i in tqdm(range(X.shape[1])):
    noise = np.random.multivariate_normal(mean, cov, X.shape[0])
    X_noisy = X + noise * dirac_mask(n_columns, i)
    is_valid = constraints_checker.check_constraints(X_noisy.values, X_noisy.values)

    X_valid = X_noisy[is_valid]
    X_invalid = X_noisy[~is_valid]

    data["valid"].append(X_valid)
    data["invalid"].append(X_invalid)

  return data


sigma = 1e3
sigma
logger.info("Generating noisy attacks on train set")
data_train = generate_noisy_attacks_by_features(X_train, sigma=sigma)
logger.info("Generating noisy attacks on test set")
data_test = generate_noisy_attacks_by_features(X_test, sigma=sigma)

val_inval_proportions = [
  [val.shape[0], inval.shape[0]]
  for val, inval in zip(data_train["valid"], data_train["invalid"])
]
df_viol = pd.DataFrame(val_inval_proportions, columns=["Valid", "Invalid"])

df_viol.plot.bar(
  stacked=True,
  title=f"Constraint Violation - {dataset_name}",
  width=1,
  edgecolor="black",
  linewidth=0.1,
  xticks=[],
)


# %% DeepOD step
df_invalid_test = pd.concat([x for x in data_test["invalid"]], axis=0)
logger.info(f"Test invalid: {df_invalid_test.shape[0]}")

is_valid = constraints_checker.check_constraints(X_train.values, X_train.values)
X_train_valid, X_train_invalid = X_train[is_valid], X_train[~is_valid]
native_invalid_prop = X_train_invalid.shape[0] / X_train.shape[0] * 100

if native_invalid_prop > 10:
  logger.warning(
    f"Native invalid proportion: {native_invalid_prop:.2f}%. Goes against assumption that data is valid."
  )


# %%
models = [
  # DeepSVDD(),
  # RDP(),
  RCA(),
  ICL(),
  SLAD(),
  # REPEN(), : Gets stuck
  # GOAD(), : Gets stuck
  # NeuTraL(), : CUDA issue
]

load_dotenv()
experiment_name = "e53.1_deepod_8"

for model in models:
  # API says: put a damn numpy array!
  model.fit(X_train.values)
  df_test = pd.concat([df_invalid_test, X_test], axis=0)

  anomaly_scores = model.decision_function(df_test.values)

  y_ctr: np.ndarray = constraints_checker.check_constraints(
    df_test.values, df_test.values
  )
  print("Constraint Violations in Combined Data:")
  print(np.unique(y_ctr, return_counts=True))

  roc_auc = roc_auc_score(y_ctr, anomaly_scores)
  print(f"ROC AUC: {roc_auc:.4f}")

  print(
    (
      f"Exp: {experiment_name}, Classifier: tabnet, Model: {model.__class__.__name__},"
      f" Dataset: {dataset_name}; Metrics: ROC AUC: {roc_auc:.4f}"
    )
  )

  mlflow_log(
    experiment_name,
    params={
      "model": model.__class__.__name__,
      "dataset": dataset_name,
      "perturbation": "random",
      "sigma": sigma,
    },
    metrics={"roc_auc": roc_auc},
  )