##%% Import data
import logging
from constraints import read_constraints, evaluate_constraints
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  roc_auc_score,
)
import numpy as np
import torch
import torchattacks
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Module
import numpy as np
from deepod.models.tabular import DeepSVDD, REPEN, RDP, RCA, GOAD, NeuTraL, ICL, SLAD
import matplotlib.pyplot as plt
from lib import load_data, download_all_data

download_all_data()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


dataset = "wids"
X, y = load_data(dataset, logger)
cnf = read_constraints(dataset)


# %% Train TabNet model
# Step 1: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)

# Convert to NumPy arrays if needed
X_train, X_test = X_train.values, X_test.values
y_train, y_test = y_train.values, y_test.values

# Step 2: Initialize TabNetClassifier
model = TabNetClassifier()

# Step 3: Train the model
model.fit(
  X_train,
  y_train,
  eval_set=[(X_test, y_test)],  # Evaluation during training
  eval_metric=["accuracy"],
  max_epochs=50,
  batch_size=256,
  patience=5,  # Early stopping if no improvement
)

# Step 4: Evaluate the model
# Predict on test data
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = (
  roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
  if len(np.unique(y)) == 2
  else None
)

print(f"Accuracy: {accuracy:.4f}")
if roc_auc:
  print(f"ROC AUC: {roc_auc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# %%
class TabNetWrapper(Module):
  """
  Wrapper to make TabNet compatible with Torchattacks.
  Directly uses the TabNet PyTorch network to maintain gradients.
  """

  def __init__(self, tabnet_model):
    super(TabNetWrapper, self).__init__()
    self.tabnet_network = tabnet_model.network  # Direct access to PyTorch network

  def forward(self, x):
    """
    Forward pass for the TabNet model.
    Only return logits (the first element of the network output).
    """
    output = self.tabnet_network(x)  # TabNet returns a tuple
    logits = output[0]  # Extract only the logits
    return logits


def generate_pgd_attack(X, y):
  # Convert test data to PyTorch tensors with requires_grad=True
  X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
  y_tensor = torch.tensor(y, dtype=torch.long)

  # Wrap test data into DataLoader for batch processing
  data_loader = DataLoader(
    TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=False
  )

  wrapped_model = TabNetWrapper(model)  # Use the fixed wrapper

  pgd_attack = torchattacks.PGD(
    model=wrapped_model,  # Wrapped TabNet model
    eps=0.3,  # Maximum perturbation
    alpha=0.01,  # Step size
    steps=40,  # Number of iterations
  )

  X_adv_list = []  # To store adversarial examples
  y_adv_list = []  # To store corresponding labels

  for X_batch, y_batch in data_loader:
    # Generate adversarial examples for each batch
    X_adv = pgd_attack(X_batch, y_batch)
    X_adv_list.append(X_adv.detach())  # Detach to avoid extra graph tracking
    y_adv_list.append(y_batch)

  # Combine batches into a single tensor
  X_adv = torch.cat(X_adv_list)
  # y_adv = torch.cat(y_adv_list)

  print("Adversarial examples crafted successfully!")
  print(f"Shape of adversarial dataset: {X_adv.shape}")
  print(f"Requires Grad: {X_adv.requires_grad}")
  return X_adv


X_adv_test = generate_pgd_attack(X_test, y_test)
X_adv_train = generate_pgd_attack(X_train, y_train)


# %% Evaluate if the PGD works
# Evaluate on original test data
y_test_pred = model.predict(X_test)  # Predictions on clean data
accuracy_clean = accuracy_score(y_test, y_test_pred)
print(f"Accuracy on Clean Test Data: {accuracy_clean:.4f}")

# Evaluate on adversarial examples
X_adv_np = X_adv_test.cpu().detach().numpy()  # Convert adversarial examples to NumPy
y_adv_pred = model.predict(X_adv_np)  # Predictions on adversarial examples
accuracy_adv = accuracy_score(y_test, y_adv_pred)
print(f"Accuracy on Adversarial Data: {accuracy_adv:.4f}")

# Compare the results
print("\nClassification Report on Adversarial Data:")
print(classification_report(y_test, y_adv_pred))


# %% Take adversarial examples and split into valid and invalid
X_adv_test_np = X_adv_test.cpu().detach().numpy()
X_adv_train_np = X_adv_train.cpu().detach().numpy()

is_valid = evaluate_constraints(pd.DataFrame(X_adv_test_np, columns=X.columns), cnf)
X_adv_test_valid = X_adv_test_np[is_valid]  # Clean
X_adv_test_invalid = X_adv_test_np[~is_valid]  # Dirty

is_valid = evaluate_constraints(pd.DataFrame(X_adv_train_np, columns=X.columns), cnf)
X_adv_train_valid = X_adv_train_np[is_valid]  # Clean
X_adv_train_invalid = X_adv_train_np[~is_valid]  # Dirty

print(
  f"Out of all the train adversarial attacks {X_adv_train_valid.shape[0]} are valid and {X_adv_train_invalid.shape[0]} are invalid"
)
print(
  f"Out of all the test adversarial attacks {X_adv_test_valid.shape[0]} are valid and {X_adv_test_invalid.shape[0]} are invalid"
)


# %% DeepOD step
# Combine clean and clean adversarial data
is_valid = evaluate_constraints(pd.DataFrame(X_train, columns=X.columns), cnf)
X_train_valid = X_train[is_valid]
X_train_invalid = X_train[~is_valid]
X_train_valid_combined = np.vstack([X_train_valid, X_adv_train_valid])
X_train_od = X_train_valid_combined  # Synonyms, btw

is_valid = evaluate_constraints(pd.DataFrame(X_test, columns=X.columns), cnf)
X_test_valid = X_test[is_valid]
X_test_invalid = X_test[~is_valid]

# Shuffle data for robustness
indices = np.arange(X_train_od.shape[0])
np.random.shuffle(indices)
X_train_od = X_train_od[indices]


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


def anomaly_hist(anomaly_scores):
  fig, ax = plt.subplots()
  ax.hist(anomaly_scores, bins=30, color="blue", edgecolor="black")
  ax.set_title("Histogram of Anomaly Scores")
  ax.set_xlabel("Anomaly Score")
  ax.set_ylabel("Frequency")
  return fig


# Combine clean and unconstrained adversarial data
X_test_invalid_combined = np.vstack([X_test_invalid, X_adv_test_invalid])
X_test_valid_combined = np.vstack([X_test_valid, X_adv_test_valid])
X_test_od = np.vstack([X_test_valid_combined, X_test_invalid_combined])

for model in models:
  # model = models[6]
  model.fit(X_train_od)
  anomaly_scores = model.decision_function(X_test_od)

  df_test_od = pd.DataFrame(X_test_od, columns=X.columns)
  y_ctr = evaluate_constraints(df_test_od, cnf)
  print("Constraint Violations in Combined Data:")
  print(y_ctr.value_counts(normalize=False))

  roc_auc = roc_auc_score(y_ctr, anomaly_scores)
  print(f"ROC AUC: {roc_auc:.4f}")

  print(
    f"Exp: e53.1_deepod_6, Classifier: tabnet, Model: {model.__class__.__name__}, Dataset: {dataset}; Metrics: ROC AUC: {roc_auc:.4f}"
  )


# %% Debug
from umap import UMAP

umap = UMAP(n_components=2)
X_clean_umap = umap.fit_transform(X_train_od)
X_adv_umap = umap.transform(X_adv_np)

# plt.scatter(X_clean_umap[:, 0], X_clean_umap[:, 1], label="Clean Data", s=0.5)
# plt.scatter(X_adv_umap[:, 0], X_adv_umap[:, 1], label="Adversarial Data", s=0.5)
# plt.legend()

# actually train umap on X_clean and X_adv_np
X_combined_umap = umap.fit_transform(X_test_od)
# plt.scatter(X_combined_umap[:, 0], X_combined_umap[:, 1], label="Combined Data", s=0.5)
# plt.legend()

# then color based on whether it's adversarial or not
y_combined = np.zeros(X_test_od.shape[0])
y_combined[X_train_od.shape[0] :] = 1
color = np.where(y_combined == 0, "blue", "red")
plt.scatter(X_combined_umap[:, 0], X_combined_umap[:, 1], c=color, s=0.3)
plt.legend()


# %%
from matplotlib import cm
from matplotlib.colors import Normalize

# Entrenar UMAP en los datos combinados
X_combined_umap = umap.fit_transform(X_test_od)

# Normalizar las puntuaciones de anomalía para asignar colores
norm = Normalize(vmin=min(anomaly_scores), vmax=max(anomaly_scores))
anomaly_colors = cm.viridis(norm(anomaly_scores))  # Usar colormap 'viridis'

# Crear la visualización con colores basados en anomaly_scores
plt.figure(figsize=(10, 7))
plt.scatter(
  X_combined_umap[:, 0],
  X_combined_umap[:, 1],
  c=anomaly_colors,  # Colorear en base a anomaly_scores
  s=5,
  label="Datos Combinados",
)
plt.colorbar(
  cm.ScalarMappable(norm=norm, cmap="viridis"), label="Puntuación de Anomalía"
)
plt.title("Proyección UMAP con Puntuaciones de Anomalía")
plt.xlabel("UMAP Dim 1")
plt.ylabel("UMAP Dim 2")
plt.show()
