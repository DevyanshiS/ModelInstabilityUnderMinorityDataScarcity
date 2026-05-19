# Experimental Results Report: Model Instability Under Minority Data Scarcity

## Overview

This report summarizes the experimental evaluation of the AMBA framework against 9 baselines across 4 datasets and 7 base classifiers.

## Overall Performance (Averaged Across all Datasets & Models)

| Technique       |   F1_mean |   ROC_AUC_mean |   G_Mean_mean |   F1_CV |   MCC_std |
|:----------------|----------:|---------------:|--------------:|--------:|----------:|
| ThresholdMoving |    0.6881 |         0.8338 |        0.7658 |  0.0475 |    0.0505 |
| TomekLinks      |    0.6788 |         0.8523 |        0.7522 |  0.0586 |    0.0525 |
| ROS             |    0.6678 |         0.85   |        0.7852 |  0.0455 |    0.0449 |
| ADASYN          |    0.6657 |         0.8486 |        0.7873 |  0.058  |    0.0554 |
| SMOTE           |    0.6642 |         0.848  |        0.7827 |  0.0702 |    0.0595 |
| Base            |    0.6542 |         0.8486 |        0.7296 |  0.0638 |    0.0493 |
| EasyEnsemble    |    0.6415 |         0.8918 |        0.828  |  0.0549 |    0.0497 |
| BalancedBagging |    0.6376 |         0.8586 |        0.7837 |  0.0571 |    0.0559 |
| AMBA            |    0.6274 |         0.8504 |        0.7508 |  0.0429 |    0.0378 |
| RUS             |    0.625  |         0.8518 |        0.7921 |  0.0615 |    0.0563 |
| ClassWeight     |    0.6244 |         0.8485 |        0.7672 |  0.0699 |    0.0555 |



## Dataset Breakdown (Mean F1 Score per Technique)

| Technique       |   diabetes |   sick |
|:----------------|-----------:|-------:|
| ADASYN          |     0.6297 | 0.7287 |
| BalancedBagging |     0.653  | 0.6017 |
| Base            |     0.5919 | 0.7633 |
| ClassWeight     |     0.5971 | 0.6699 |
| EasyEnsemble    |     0.6503 | 0.6327 |
| ROS             |     0.6325 | 0.7294 |
| RUS             |     0.6426 | 0.5941 |
| SMOTE           |     0.6252 | 0.7326 |
| ThresholdMoving |     0.6426 | 0.7941 |
| TomekLinks      |     0.6216 | 0.7789 |
| AMBA            |     0.6026 | 0.6853 |



## Stability Analysis

A key contribution of AMBA is improving the stability of models under heavy class imbalance. The following table showcases the Coefficient of Variation (CV) for F1 Score (Lower is better).

| Technique       |   F1_CV |   Recall_CV |
|:----------------|--------:|------------:|
| AMBA            |  0.0429 |      0.0486 |
| ROS             |  0.0455 |      0.0542 |
| ThresholdMoving |  0.0475 |      0.0687 |
| EasyEnsemble    |  0.0549 |      0.0293 |
| BalancedBagging |  0.0571 |      0.05   |
| ADASYN          |  0.058  |      0.0579 |
| TomekLinks      |  0.0586 |      0.0751 |
| RUS             |  0.0615 |      0.0482 |
| Base            |  0.0638 |      0.0935 |
| ClassWeight     |  0.0699 |      0.0803 |
| SMOTE           |  0.0702 |      0.0834 |



## Visualizations

The following visualizations have been generated and saved to the `outputs/` directory:

- `f1_heatmap.png`: Mean F1 Score across all Datasets (Models vs Techniques)

- `stability_profile_cv_f1.png`: Boxplot showing the distribution of F1 Coefficient of Variation for all techniques.

- `dataset_breakdown.png`: Bar chart comparing Base vs AMBA performance per dataset.


## Conclusion

Based on the aggregated metrics above, review the performance of `AMBA` in relation to standard techniques like `SMOTE`, `BalancedBagging`, and `ThresholdMoving`. If AMBA consistently ranks in the top tier for F1 and G-Mean while maintaining a low CV, the research hypothesis stated in the proposal is strongly supported.