from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_constraint_violation_by_feature(
  val_inval_proportions: Dict[str, List[int]], dataset_name: str
):
  df_viol = pd.DataFrame(val_inval_proportions, columns=["Valid", "Invalid"])

  df_viol.plot.bar(
    stacked=True,
    title=f"Constraint Violation - {dataset_name}",
    width=1,
    edgecolor="black",
    linewidth=0.1,
    xticks=[],
  )


def plot_anomaly_scores(anomaly_scores, y_is_invalid):
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
