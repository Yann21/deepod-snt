# %%
import pandas as pd
from sklearn.model_selection import train_test_split

from deepod_snt.utils.utils import (
  downsample,
)

from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.datasets import dataset_factory

dataset_name = "lcld_v2_iid"
dataset = dataset_factory.get_dataset(dataset_name)
constraints_checker = ConstraintChecker(dataset.get_constraints(), tolerance=1e-3)

X, y = dataset.get_x_y()
X_down, y_down = downsample(X, y, frac=0.1)
X_train, X_test, y_train, y_test = train_test_split(
  X_down, y_down, test_size=0.2, random_state=42
)


# %%
import numpy as np
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

# Define only addition and subtraction as allowed functions
add = make_function(function=np.add, name="add", arity=2)
sub = make_function(function=np.subtract, name="sub", arity=2)

est = SymbolicRegressor(
  population_size=500,
  generations=10,
  stopping_criteria=0.001,
  function_set=[add, sub],
  max_samples=1.0,
  verbose=1,
  parsimony_coefficient=0.3,  # Increased penalty for complexity
  random_state=0,
)

XX = X_down[["open_acc", "total_acc"]].copy()
XX["y"] = XX["total_acc"] - XX["open_acc"]
est.fit(XX[["open_acc", "total_acc"]], XX["y"])
print(est._program)

# %%
from gplearn.genetic import SymbolicTransformer

# Extract the feature pair
X_unsup = X_down[["pub_rec_bankruptcies", "pub_rec"]].values

# Define the Symbolic Transformer (to discover expressions)
gp = SymbolicTransformer(
  population_size=1000,
  generations=20,
  function_set=["add", "sub", "mul", "div"],  # Allow all ops
  metric="pearson",  # Look for correlated expressions
  verbose=1,
  random_state=0,
)

# Fit to the data (unsupervised)
gp.fit(X_unsup, np.ones(X_unsup.shape[0]))  # Dummy target

#%%
# Print top symbolic expressions
for expr in gp:
  print(expr)


#%%
X[['pub_rec_bankruptcies', 'pub_rec']].corr()
