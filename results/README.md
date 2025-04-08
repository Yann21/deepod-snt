*CHANGELOG*
- e53.1_deepod_1: initial experiment
- e53.1_deepod_2: use anomaly_scores for ROCAUC metric instead of the previous
  hard labels
- e53.1_deepod_3: update protocol as follows: instead of trying to see if the OD
  model can identify adversarials, we instead want to see if it can predict which
  data points are constraint-respecting and which aren't. To do that we train
  the OD on a mix of valid data from the original dataset and synthetic
  adversarials and test on a combination of both valid and invalid.
  - _4: log histogram of anomaly scores to debug sub 0.5 ROC AUC 1
  - _5: redefine protocol to properly split train and test when crafting PGD attacks
  - _6: went a bit fast
  - _7: use noise algorithm as data augmentation
  - _8: use tabularbench constraint checker
  - _9: extension with simpler OD models
