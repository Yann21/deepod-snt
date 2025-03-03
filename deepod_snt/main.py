# %% Import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Dict, List
from factories import AnomalyDetectorFactory

from viz import plot_constraint_violation_by_feature
from tqdm import tqdm
from utils import downsample, dirac_mask, mlflow_log, confirm_base_data_is_valid
from dotenv import load_dotenv

from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.datasets import dataset_factory
from utils import create_loger
from sklearn.preprocessing import StandardScaler

load_dotenv()

logger = create_loger()


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


# %%
debug = False
experiment_name = "e53.1_deepod_9"
models = [
  "autoencoder",
  "oneclasssvm",
  "isolationforest",
  # "deepsvdd",
  # "rdp",
  # "rca",
  # "icl",
  # "slad",
  # "repen",  # : Gets stuck
  # "goad",  # : Gets stuck
  # "neutral",  # : CUDA issue
]
dataset_names = [
  # "ctu_13_neris",
  "lcld_v2_iid",
]
sigmas = [1e3]
sample = 0.1

for dataset_name, sigma in zip(dataset_names, sigmas):
  dataset = dataset_factory.get_dataset(dataset_name)
  constraints_checker = ConstraintChecker(dataset.get_constraints(), tolerance=1e-3)

  X, y = dataset.get_x_y()
  X_down, y_down = downsample(X, y, frac=sample)
  X_train, X_test, y_train, y_test = train_test_split(
    X_down, y_down, test_size=0.2, random_state=42
  )

  logger.info("Generating noisy attacks on train set")
  data_train = generate_noisy_attacks_by_features(X_train, sigma=sigma)
  logger.info("Generating noisy attacks on test set")
  data_test = generate_noisy_attacks_by_features(X_test, sigma=sigma)

  val_inval_proportions = [
    [val.shape[0], inval.shape[0]]
    for val, inval in zip(data_train["valid"], data_train["invalid"])
  ]
  plot_constraint_violation_by_feature(val_inval_proportions, dataset_name)

  df_invalid_test = pd.concat([x for x in data_test["invalid"]], axis=0)
  logger.info(f"Test invalid: {df_invalid_test.shape[0]}")
  confirm_base_data_is_valid(constraints_checker, X_train, logger)

  for model in models:
    params = {}
    if model == "oneclasssvm":
      params["kernel"] = "linear"
    model = AnomalyDetectorFactory.create(model, **params)

    if model == "autoencoder":
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)
      df_invalid_test_scaled = scaler.transform(df_invalid_test)

      model.fit(X_train_scaled.values)
      df_test = pd.concat([df_invalid_test_scaled, X_test_scaled], axis=0)
    else:
      model.fit(X_train.values)
      df_test = pd.concat([df_invalid_test, X_test], axis=0)

    anomaly_scores = model.decision_function(df_test.values)

    y_is_invalid: np.ndarray = ~constraints_checker.check_constraints(
      df_test.values, df_test.values
    )
    logger.info("Constraint Violations in Combined Data:")
    logger.info(np.unique(y_is_invalid, return_counts=True))

    roc_auc = roc_auc_score(y_is_invalid, anomaly_scores)
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(
      (
        f"Exp: {experiment_name}, Classifier: tabnet, Model: {model},"
        f" Dataset: {dataset_name}; Metrics: ROC AUC: {roc_auc:.4f}"
      )
    )

    # mlflow_log(
    #   experiment_name,
    #   params={
    #     "model": model,
    #     "dataset": dataset_name,
    #     "perturbation": "random",
    #     "sigma": sigma,
    #     "debug": debug,
    #   },
    #   metrics={"roc_auc": roc_auc},
    # )


# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(
  anomaly_scores[y_is_invalid == 1], label="Valid", color="blue", kde=True, alpha=0.5
)
sns.histplot(
  anomaly_scores[y_is_invalid == 0], label="Invalid", color="red", kde=True, alpha=0.5
)
plt.legend()
plt.xlabel("Anomaly Score")
plt.ylabel("Density")
plt.title("Distribution of Anomaly Scores")
plt.show()
