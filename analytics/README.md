# Analytics

Run `python 01_eda.py` and then `python 02_modeling.py`. The first script performs the single `sns.load_dataset('titanic')` load and commits `titanic.csv` as the offline fallback. The second script reads the same cleaned CSV.

The EDA report contains required missingness percentages, threshold decisions, IQR outlier counts, fare mean/median/mode and skew conclusion, the required six-column correlation analysis, two strongest correlations, four-plus chart interpretations, bivariate survival rates, and standardization sanity check. The modeling report contains the stratified split, leakage-safe preprocessing, three classifiers and metrics, imbalance comparison, RF GridSearchCV/OOB result, regression metrics/residual discussion, final recommendation, and reload test for the complete joblib pipeline.
