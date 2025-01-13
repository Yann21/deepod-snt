#%%
import pandas as pd
import gdown
from pathlib import Path

here = Path(__file__).parent

def download_all_data() -> None:
  folder = "https://drive.google.com/drive/folders/12xtPwtW5sZUxe_hxWKG-U_zdNwuYbF8q?usp=drive_link"
  gdown.download_folder(folder)

def load_data(dataset, logger):
  """Load dataset from file."""
  logger.info(f"Loading data for dataset: {dataset}")
  df = pd.read_csv(here / f"data/constrained/{dataset}/data.csv")
  return df.drop(columns=["target"]), df["target"]