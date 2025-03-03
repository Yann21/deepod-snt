from deepod.models.tabular import DeepSVDD, REPEN, RDP, RCA, GOAD, NeuTraL, ICL, SLAD
from abc import ABC, abstractmethod
from pyod.models.auto_encoder import AutoEncoder
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


class BaseAnomalyDetector(ABC):
  @abstractmethod
  def fit(self, data):
    pass

  @abstractmethod
  def decision_function(self, data):
    pass


class AnomalyDetectorFactory:
  _models = {
    "deepsvdd": DeepSVDD,
    "oneclasssvm": OneClassSVM,
    "isolationforest": IsolationForest,
    "rdp": RDP,
    "rca": RCA,
    "icl": ICL,
    "slad": SLAD,
    "repen": REPEN,  # : Gets stuck
    "goad": GOAD,  # : Gets stuck
    "neutral": NeuTraL,  # : CUDA issue
    "autoencoder": AutoEncoder,
  }

  @staticmethod
  def create(model_name: str, **kwargs) -> BaseAnomalyDetector:
    if model_name in AnomalyDetectorFactory._models:
      return AnomalyDetectorFactory._models[model_name](**kwargs)
    else:
      raise ValueError(f"Unknown model: {model_name}")


#%%
