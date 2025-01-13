import pandas as pd
from typing import List
import re
import numpy as np
from typing import Dict
import sympy as sp
from pathlib import Path
from tqdm import tqdm

class CNF:
  def __init__(self, clauses: List[sp.Expr]):
    self.clauses = clauses

  def __repr__(self):
    return f"AND({self.clauses})"

  def __getitem__(self, index):
    return self.clauses[index]

  def __len__(self):
    return len(self.clauses)

  def __iter__(self):
    return iter(self.clauses)



def find_constraint_closures(cnf: CNF) -> Dict[sp.Symbol, List[sp.Symbol]]:
  """
  From a CNF expression, find all variable closures based on their connections through clauses.

  Args:
  cnf (CNF): CNF object containing a list of clauses.

  Returns:
  Dict[sp.Symbol, List[sp.Symbol]]: Dictionary mapping each symbol to a list of all symbols in its closure.

  Example:
  >>> cnf = CNF([a & b, b & ~c])
  >>> find_constraint_closures(cnf)
  {a: [a, b, c], b: [a, b, c], c: [a, b, c]}
  """
  assert isinstance(cnf, CNF)
  clauses = cnf.clauses
  associations = {}

  def find_group(symbol, visited):
    """Find the full group for a symbol, avoiding cycles."""
    if symbol in visited:
      return set()
    visited.add(symbol)
    group = set(associations[symbol])
    for sym in associations[symbol]:
      group.update(find_group(sym, visited))
    return group

  # Initialize the associations
  for clause in clauses:
    symbols = list(clause.atoms(sp.Symbol))
    for sym in symbols:
      if sym not in associations:
        associations[sym] = set()
      associations[sym].update(symbols)

  # Resolve full associations
  full_associations = {}
  for sym in associations:
    full_group = find_group(sym, set())
    full_associations[sym] = sorted(list(full_group), key=lambda x: str(x))

  return full_associations


def expand_constraints_to_features(
  constraints_violation_matrix: np.ndarray,
  cnf: CNF,
  closures: Dict[sp.Symbol, List[sp.Symbol]],
) -> np.ndarray:
  """
  Transform the constraints matrix into a feature matrix based on the constraint closures.

  Args:
    constraints_violation_matrix (np.ndarray): The original constraints violation matrix (rows, constraints).
    cnf (CNF): CNF object containing a list of clauses.
    closures (Dict[sp.Symbol, List[sp.Symbol]]): The closure mapping for each feature.

  Returns:
    np.ndarray: Transformed matrix (rows, features).
  """
  assert isinstance(cnf, CNF)
  assert isinstance(constraints_violation_matrix, np.ndarray)
  clauses = cnf.clauses

  num_rows, num_constraints = constraints_violation_matrix.shape
  feature_symbols = list(set().union(*closures.values()))
  feature_indices = {sym: idx for idx, sym in enumerate(feature_symbols)}
  num_features = len(feature_symbols)
  feature_matrix = np.zeros((num_rows, num_features), dtype=int)

  for row in range(num_rows):
    for constraint_idx, clause in enumerate(clauses):
      if constraints_violation_matrix[row, constraint_idx]:
        involved_symbols = list(clause.atoms(sp.Symbol))
        for symbol in involved_symbols:
          for closure_symbol in closures[symbol]:
            feature_matrix[row, feature_indices[closure_symbol]] = 1

  return feature_matrix


def get_feature_violation(df: pd.DataFrame, cnf: CNF) -> pd.DataFrame:
  """
  Get a mask of features that violate any constraints.

  Args:
      df (pd.DataFrame): The input dataframe.
      cnf (CNF): CNF expression representing all constraints.

  Returns:
      pd.DataFrame: A mask of features that violate the constraints. (rows, features)
  """
  # Evaluate each clause in CNF and find variable closures
  clause_satisfaction = evaluate_clause_by_clause(df, cnf)
  constraint_violation = ~clause_satisfaction

  closures = find_constraint_closures(cnf)

  # Convert closures mapping from symbols to indices
  feature_violation_matrix = expand_constraints_to_features(
    constraint_violation.values, cnf, closures
  )

  # Adjust matrix size if needed to match DataFrame columns
  padded_feature_violation = np.pad(
    feature_violation_matrix,
    ((0, 0), (0, df.shape[1] - feature_violation_matrix.shape[1])),
    mode="constant",
  )

  return pd.DataFrame(padded_feature_violation, columns=df.columns)



def monte_carlo_hausdorff_measure(constraints: List[str]):
  """Approximation of the hausdorff measure of an open geometry."""
  points = np.random.uniform(-1e12, 1e12, (int(1e6), 100))

  df_ = pd.DataFrame(points)
  df_filtered = filter_by_constraints(df_, constraints)

  return df_filtered.shape[0] / 1e6 * 100


here = Path(__file__).parent
tqdm.pandas()



def extract_number(s: str) -> int:
  """
  >>> extract_number("y_27")
  27
  """
  match = re.search(r"y_(\d+)", s)
  assert match, f"Column not of the form y_(\d+)"
  return int(match.group(1))


def get_column_index(var: str, columns: pd.Index) -> int:
  """
  >>> get_column("bob", pd.Index(["alice", "bob", "charlie"])
  1
  """
  var = str(var)
  if var in columns:
    return columns.get_loc(var)
  else:
    return extract_number(var)


def evaluate_constraints(df: pd.DataFrame, cnf: CNF) -> pd.Series:
  """Evaluate the CNF expression on the DataFrame and return a mask of satisfaction."""
  variables = list(set().union(*[clause.free_symbols for clause in cnf.clauses]))
  columns = df.columns
  variable_indices = {var: get_column_index(var, columns) for var in variables}

  # Create a dictionary for lambdify
  lambdify_dict = {
    var: sp.Symbol(f"col_{get_column_index(var, columns)}") for var in variables
  }

  # Combine all clauses into a single expression for evaluation
  combined_cnf = sp.And(*cnf.clauses)

  # Lambdify the combined CNF expression with NumPy as the module
  cnf_func = sp.lambdify(
    [lambdify_dict[var] for var in variables],
    combined_cnf.subs(lambdify_dict),
    modules="numpy",
  )

  # Apply the function to the DataFrame columns
  mask = cnf_func(*[df.iloc[:, variable_indices[var]].values for var in variables])

  return pd.Series(mask, index=df.index)


def evaluate_clause_by_clause(df: pd.DataFrame, cnf: CNF) -> pd.DataFrame:
  """Evaluate each clause in the CNF separately and return results in a DataFrame."""
  clauses = cnf.clauses
  results = {}

  # Process each clause
  for i, clause in enumerate(clauses):
    # Extract variables and prepare for substitution and evaluation
    variables = list(clause.free_symbols)
    columns = df.columns
    variable_indices = {var: get_column_index(var, columns) for var in variables}

    # Dictionary to replace symbols with column-based symbols for lambdify
    lambdify_dict = {
      var: sp.Symbol(f"col_{variable_indices[var]}") for var in variables
    }

    # Lambdify the current clause
    clause_func = sp.lambdify(
      [lambdify_dict[var] for var in variables],
      clause.subs(lambdify_dict),
      modules="numpy",
    )

    # Evaluate the clause on the DataFrame
    results[str(clause)] = clause_func(
      *[df.iloc[:, variable_indices[var]].values for var in variables]
    )

  # Create and return the results DataFrame
  return pd.DataFrame(results, index=df.index)


def serialize_cnf(constraints: List[sp.Expr]) -> str:
  """Unfortunately, sympy simplifies CNF so we need to serialize independent terms
  of the CNF to recover the actual form."""
  return "\n".join([sp.srepr(c) for c in constraints])


def deserialize_cnf(serialized_cnf: str) -> CNF:
  """Deserialize a serialized CNF."""
  # We have a pre-commit hook that adds newlines and since I'm lazy I'll remove it here
  xs = [x for x in serialized_cnf.split("\n") if x]
  return CNF([sp.sympify(c) for c in xs])

def read_constraints(dataset: str) -> CNF:
  with open(here / f"../../data/constrained/{dataset}/constraints.sp") as f:
    constraints_str = f.read()
    cnf = deserialize_cnf(constraints_str)
  return cnf

def encode_feature_violation(df: pd.DataFrame, cnf: CNF) -> pd.DataFrame:
  feature_violation = get_feature_violation(df, cnf)
  feature_violation.columns = [
    "is_violating_" + str(col) for col in feature_violation.columns
  ]
  return pd.concat([df.reset_index(drop=True).copy(), feature_violation], axis=1)
