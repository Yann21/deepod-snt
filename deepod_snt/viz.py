from typing import Dict, List
import pandas as pd


def plot_constraint_violation_by_feature(val_inval_proportions: Dict[str, List[int]], dataset_name: str):
  df_viol = pd.DataFrame(val_inval_proportions, columns=["Valid", "Invalid"])

  df_viol.plot.bar(
    stacked=True,
    title=f"Constraint Violation - {dataset_name}",
    width=1,
    edgecolor="black",
    linewidth=0.1,
    xticks=[],
  )
