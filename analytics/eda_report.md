# EDA Report

## Shape
`(891, 15)`

## Missing percentages before cleaning
|             |         0 |
|:------------|----------:|
| age         | 19.8653   |
| embarked    |  0.224467 |
| deck        | 77.1044   |
| embark_town |  0.224467 |

## Cleaning decisions
Columns under 5% missing had affected rows dropped; columns from 5% through 30% were imputed (numeric median, categorical mode). Columns above 30% were retained as an explicit `Missing` category. This follows the required threshold rule and avoids unreliable numeric imputation.

## Outliers and fare shape
Age IQR outliers: **65**. Fare IQR outliers: **114**. Fare mean=32.097, median=14.454, mode=8.050; because mean > median > mode, fare is **right-skewed**.

## Survival rates
### Sex
| sex    |   survival_rate |
|:-------|----------------:|
| female |        0.740385 |
| male   |        0.188908 |

### Pclass
|   pclass |   survival_rate |
|---------:|----------------:|
|        1 |        0.626168 |
|        2 |        0.472826 |
|        3 |        0.242363 |

### Sex + Pclass
|               |   survival_rate |
|:--------------|----------------:|
| ('female', 1) |        0.967391 |
| ('female', 2) |        0.921053 |
| ('female', 3) |        0.5      |
| ('male', 1)   |        0.368852 |
| ('male', 2)   |        0.157407 |
| ('male', 3)   |        0.135447 |

## Correlation matrix
|          |   survived |     pclass |        age |      sibsp |      parch |       fare |
|:---------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| survived |  1         | -0.335549  | -0.0698217 | -0.03404   |  0.0831508 |  0.25529   |
| pclass   | -0.335549  |  1         | -0.336512  |  0.0816556 |  0.0168245 | -0.548193  |
| age      | -0.0698217 | -0.336512  |  1         | -0.232543  | -0.171485  |  0.0937071 |
| sibsp    | -0.03404   |  0.0816556 | -0.232543  |  1         |  0.414542  |  0.160887  |
| parch    |  0.0831508 |  0.0168245 | -0.171485  |  0.414542  |  1         |  0.217532  |
| fare     |  0.25529   | -0.548193  |  0.0937071 |  0.160887  |  0.217532  |  1         |

The two strongest absolute off-diagonal correlations are **pclass vs fare (-0.548)** and **sibsp vs parch (0.415)**. The first reflects the strongest linear association among the required numeric variables; the second is the next strongest and should be interpreted as association, not causation.

## Multivariate data story
1. **Survival by sex:** females have a substantially higher survival rate than males, making sex one of the clearest group-level separators.
2. **Survival by class:** survival generally decreases as passenger class moves from first to third, indicating socioeconomic/class differences in outcomes.
3. **Sex and class together:** the combined chart shows that both variables matter; within classes, women generally survive at higher rates, while third-class outcomes are weaker.
4. **Age/fare distributions:** age shows the passenger age structure while fare is strongly right-skewed, with a small number of very high fares.

## Standardization check
|        |   raw_mean |   raw_std |        z_mean |   z_std |
|:-------|-----------:|----------:|--------------:|--------:|
| age    |    29.3152 |   12.9776 | nan           |     nan |
| age_z  |   nan      |  nan      |   2.71749e-16 |       1 |
| fare   |    32.0967 |   49.6695 | nan           |     nan |
| fare_z |   nan      |  nan      |   1.39871e-16 |       1 |

The z-score columns have approximately mean 0 and population standard deviation 1, confirming the exploratory transformation. This standardized data is not used by the modeling pipeline; modeling performs train-only preprocessing.
