# %% Import data
import logging
from constraints import read_constraints, evaluate_constraints
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Dict, List
import mlflow

from deepod.models.tabular import DeepSVDD, REPEN, RDP, RCA, GOAD, NeuTraL, ICL, SLAD
import matplotlib.pyplot as plt
from utils import load_data
from umap import UMAP
from matplotlib import cm
from matplotlib.colors import Normalize
from tqdm import tqdm
from utils import downsample, dirac_mask
from constraints import (
  evaluate_constraints,
  CNF,
  get_variables_from_clauses,
  get_feature_violation,
  evaluate_clause_by_clause,
)


# %%

# download_all_data()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# dataset = "lcld_v2_iid"
# dataset = "ctu_13_neris"
dataset = "wids"
# dataset = "heloc_linear"
X, y = load_data(dataset, logger)
cnf = read_constraints(dataset)
X_down, y_down = downsample(X, y, frac=1)


# %%
# is_valid = evaluate_constraints(X_down, cnf)
# feature_violation = get_feature_violation(X_down, cnf)
# clause_violation = ~evaluate_clause_by_clause(X_down, cnf)


# %%
if dataset == "lcld_v2_iid":
  logger.info("Converting issue_d to int")
  X_down["issue_d"] = pd.to_datetime(X_down["issue_d"]).astype("int")

X_train, X_test, y_train, y_test = train_test_split(
  X_down, y_down, test_size=0.2, random_state=42
)


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
    is_valid = evaluate_constraints(X_noisy, cnf)

    X_valid = X_noisy[is_valid]
    X_invalid = X_noisy[~is_valid]

    data["valid"].append(X_valid)
    data["invalid"].append(X_invalid)

  return data


logger.info("Generating noisy attacks on train set")
data_train = generate_noisy_attacks_by_features(X_train)
logger.info("Generating noisy attacks on test set")
data_test = generate_noisy_attacks_by_features(X_test)

val_inval_proportions = [
  [val.shape[0], inval.shape[0]]
  for val, inval in zip(data_train["valid"], data_train["invalid"])
]
df_viol = pd.DataFrame(val_inval_proportions, columns=["Valid", "Invalid"])

df_viol.plot.bar(
  stacked=True,
  title=f"Constraint Violation - {dataset}",
  width=1,
  edgecolor="black",
  linewidth=0.1,
  xticks=[],
)


# %%
df_invalid_test = pd.concat([x for x in data_test["invalid"]], axis=0)
# df_valid_train = pd.concat([x for x in data_train["valid"]], axis=0)
# df_valid_train = df_valid_train.sample(n=10000, replace=False)

# print(f"Train valid: {df_valid_train.shape[0]}")
print(f"Test invalid: {df_invalid_test.shape[0]}")


# %% DeepOD step
is_valid = evaluate_constraints(X_train, cnf)
X_train_valid, X_train_invalid = X_train[is_valid], X_train[~is_valid]
native_invalid_prop = X_train_invalid.shape[0] / X_train.shape[0] * 100

if native_invalid_prop > 10:
  logger.warning(
    f"Native invalid proportion: {native_invalid_prop:.2f}%. Goes against assumption that data is valid."
  )
else:
  logger.info(f"Native invalid proportion: {native_invalid_prop:.2f}%")


# %%
# Shuffle data for robustness
# indices = np.arange(X_train_od.shape[0])
# np.random.shuffle(indices)
# X_train_od = X_train_od[indices]


models = [
  DeepSVDD(),
  REPEN(),
  RDP(),
  RCA(),
  GOAD(),
  NeuTraL(),
  ICL(),
  SLAD(),
]

# def anomaly_hist(anomaly_scores):
#   fig, ax = plt.subplots()
#   ax.hist(anomaly_scores, bins=30, color="blue", edgecolor="black")
#   ax.set_title("Histogram of Anomaly Scores")
#   ax.set_xlabel("Anomaly Score")
#   ax.set_ylabel("Frequency")
#   return fig


# %%
import os
from load_dotenv import load_dotenv

load_dotenv()
experiment_name = "e53.1_deepod_7"

for model in models:
  # API says: put a damn numpy array!
  model.fit(X_train.values)
  df_test = pd.concat([df_invalid_test, X_test], axis=0)

  anomaly_scores = model.decision_function(df_test.values)

  y_ctr = evaluate_constraints(df_test, cnf)
  print("Constraint Violations in Combined Data:")
  print(y_ctr.value_counts(normalize=False))

  roc_auc = roc_auc_score(y_ctr, anomaly_scores)
  print(f"ROC AUC: {roc_auc:.4f}")

  print(
    (
      f"Exp: {experiment_name}, Classifier: tabnet, Model: {model.__class__.__name__},"
      f" Dataset: {dataset}; Metrics: ROC AUC: {roc_auc:.4f}"
    )
  )

  mlflow.set_tracking_uri(
    (
      f"http://{os.environ['MLFLOW_USERNAME']}:{os.environ['MLFLOW_PASSWORD']}-"
      "@{os.environ['MLFLOW_ADDRESS']}"
    )
  )
  mlflow.set_experiment(experiment_name)
  with mlflow.start_run():
    mlflow.log_params(
      {
        "model": model.__class__.__name__,
        "dataset": dataset,
      }
    )
    mlflow.log_metrics(
      {
        "roc_auc": roc_auc,
      }
    )
