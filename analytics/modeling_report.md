# Modeling Report

## Split and class balance
The split is stratified so the train and test sets preserve the observed survived/not-survived proportions, reducing evaluation variance from class composition differences.

Class balance:
|   survived |   count |
|-----------:|--------:|
|          0 |     549 |
|          1 |     340 |

## Classifier comparison
| model               |   accuracy |   precision |   recall |       f1 |      auc |
|:--------------------|-----------:|------------:|---------:|---------:|---------:|
| Logistic Regression |   0.808989 |    0.783333 | 0.691176 | 0.734375 | 0.860963 |
| Decision Tree       |   0.764045 |    0.76     | 0.558824 | 0.644068 | 0.837366 |
| Random Forest       |   0.808989 |    0.765625 | 0.720588 | 0.742424 | 0.819586 |

Each model uses the identical split and a training-only preprocessing pipeline. Confusion matrices were computed for each model in the execution loop; ROC curves and AUC are in `artifacts/roc_comparison.png`.

## Imbalance comparison
| strategy              |   precision |   recall |       f1 |
|:----------------------|------------:|---------:|---------:|
| baseline              |    0.783333 | 0.691176 | 0.734375 |
| class_weight_balanced |    0.71831  | 0.75     | 0.733813 |
| SMOTE                 |    0.735294 | 0.735294 | 0.735294 |

SMOTE is applied only to the training data inside an imbalanced-learn pipeline. The preferred strategy is the row with the strongest F1 while maintaining an acceptable precision/recall balance.

## Random Forest GridSearchCV
Best parameters: `{'model__max_depth': 10, 'model__max_features': 'sqrt', 'model__n_estimators': 200}`. The final OOB-enabled Random Forest was refit with those parameters and its OOB score is **0.8200**.

## Regression
MAE=21.1386, RMSE=41.7465, R²=0.3468, Adjusted R²=0.3118. The residual plot is in `artifacts/residuals.png`. Heteroscedasticity is assessed visually: a clear funnel/non-random spread would indicate it; the plot should be inspected alongside the reported metrics.

## Final recommendation
Based on held-out F1, the recommended classifier is **Random Forest** (F1=0.742, AUC=0.820). It is selected because F1 balances precision and recall for the survival target. The decision should also consider recall if missing survivors is more costly than false positives. The saved `best_pipeline.joblib` contains preprocessing plus estimator and can accept raw feature rows. Reload validation: **True**.
