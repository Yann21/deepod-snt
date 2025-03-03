import pandas as pd
from typing import List
import re
import numpy as np
from typing import Dict
import sympy as sp
from pathlib import Path
from tqdm import tqdm
from typing import Union
import numpy as np
import pandas as pd
import sympy as sp
from sympy.core.symbol import Symbol
from sympy.core.relational import Relational
from typing import List
import sympy as sp
from time import perf_counter


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

  def free_symbols(self) -> set:
    symbols: List[set] = [clause.free_symbols for clause in self.clauses]
    return set.union(*symbols)


def get_variables_from_clauses(clauses: List[Relational]) -> List[Symbol]:
  """Get all the variables in a list of clauses. Same function in CNF."""
  return list(set().union(*[clause.free_symbols for clause in clauses]))


def _get_violating_features_from_clause_violation(
  row: pd.Series, cnf: CNF
) -> List[Symbol]:
  """Map the clauses that are violated to the features that violate each clause.

  >>> print(cnf[0])
  x1 & x2
  >>> print(cnf[1])
  x4 & x5
  >>> clause_violation = pd.Series([1, 1, 0, 0, 0])
  >>> assert len(clause_violation) == 7
  >>> get_violating_features_from_clause_violation(clause_violation, cnf)
  [x1, x2, x4, x5]
  """
  violating_clauses_idx: List[int] = list(np.where(row)[0])
  clauses = cnf.clauses
  violating_clauses = np.array(clauses)[violating_clauses_idx]

  return get_variables_from_clauses(violating_clauses)


def get_feature_violation(df, cnf: CNF):
  all_symbols = cnf.free_symbols()

  def find_violating_features(row):
    """Find the features that violate and place them in a list with all the symbols in the CNF."""
    violating_features = _get_violating_features_from_clause_violation(row, cnf)
    return pd.Series(
      [symbol in violating_features for symbol in all_symbols], index=all_symbols
    )

  clause_satisfaction = evaluate_clause_by_clause(df, cnf)
  clause_violation = ~clause_satisfaction
  return clause_violation.apply(find_violating_features, axis=1)


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


# Global cache for CNF substitutions
subs_cache = {}

def get_cached_subs(combined_cnf, lambdify_dict):
    """Cache CNF substitutions to avoid recomputation.
    Workaround for multiple calls to the substitute function in CNF.
    """
    cnf_key = str(combined_cnf)  # Convert CNF expression to a unique key
    
    if cnf_key in subs_cache:
        return subs_cache[cnf_key]  # Return cached substituted expression
    
    # Perform substitution (slow step)
    substituted_cnf = combined_cnf.subs(lambdify_dict)
    
    # Store in cache
    subs_cache[cnf_key] = substituted_cnf
    return substituted_cnf



def evaluate_constraints(df: pd.DataFrame, cnf: Union[CNF, str]) -> pd.Series:
  """Evaluate the CNF expression on the DataFrame and return a mask of satisfaction.
  args:
    cnf (Union[CNF, str]): The CNF expression to evaluate. Can be a CNF object
    or a string used by DataFrame.query().
  """
  if isinstance(cnf, str):
    try:
      # Return a mask of rows that satisfy the constraint
      return df.index.isin(df.query(cnf).index)
    except Exception as e:
      raise ValueError(f"Invalid string constraint: {cnf}") from e
  
  variables = list(set().union(*[clause.free_symbols for clause in cnf.clauses]))
  columns = df.columns
  variable_indices = {var: get_column_index(var, columns) for var in variables}

  # Create a dictionary for lambdify
  lambdify_dict = {
    var: sp.Symbol(f"col_{get_column_index(var, columns)}") for var in variables
  }

  # Combine all clauses into a single expression for evaluation
  combined_cnf = sp.And(*cnf.clauses)

  substituted_cnf = get_cached_subs(combined_cnf, lambdify_dict)
  # Lambdify the combined CNF expression with NumPy as the module
  cnf_func = sp.lambdify(
    [lambdify_dict[var] for var in variables],
    substituted_cnf,
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
  with open(here / f"data/constrained/{dataset}/constraints.sp") as f:
    constraints_str = f.read()
    cnf = deserialize_cnf(constraints_str)
  return cnf


def encode_feature_violation(df: pd.DataFrame, cnf: CNF) -> pd.DataFrame:
  feature_violation = get_feature_violation(df, cnf)
  feature_violation.columns = [
    "is_violating_" + str(col) for col in feature_violation.columns
  ]
  return pd.concat([df.reset_index(drop=True).copy(), feature_violation], axis=1)
