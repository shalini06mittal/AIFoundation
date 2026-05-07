# LAB DOCUMENT
## Machine Learning with Scikit-learn
*Building, Training, and Evaluating ML Models*

| **Course** | Applied Machine Learning |
|---|---|
| **Lab Module** | Scikit-learn – ML Models |
| **Difficulty** | Intermediate |
| **Duration** | 3–4 Hours |
| **Prerequisites** | Python, NumPy, Pandas basics |

Version 1.0  |  2025

---

## Table of Contents

1. [Introduction to Scikit-learn](#1-introduction-to-scikit-learn)
   - 1.1 [Why Scikit-learn?](#11-why-scikit-learn)
   - 1.2 [Installation & Setup](#12-installation--setup)
   - 1.3 [The Scikit-learn API Design](#13-the-scikit-learn-api-design)
2. [Datasets & Data Preparation](#2-datasets--data-preparation)
   - 2.1 [Built-in Datasets](#21-built-in-datasets)
   - 2.2 [Loading and Exploring Data](#22-loading-and-exploring-data)
   - 2.3 [Train / Test Split](#23-train--test-split)
   - 2.4 [Feature Preprocessing](#24-feature-preprocessing)
3. [Classification Algorithms](#3-classification-algorithms)
   - 3.1 [Logistic Regression](#31-logistic-regression)
   - 3.2 [K-Nearest Neighbours (KNN)](#32-k-nearest-neighbours-knn)
   - 3.3 [Support Vector Machine (SVM)](#33-support-vector-machine-svm)
   - 3.4 [Decision Tree](#34-decision-tree)
   - 3.5 [Random Forest](#35-random-forest)
   - 3.6 [Gradient Boosting](#36-gradient-boosting)
4. [Regression Algorithms](#4-regression-algorithms)
   - 4.1 [Setup for Regression](#41-setup-for-regression)
   - 4.2 [Linear Regression](#42-linear-regression)
   - 4.3 [Regression Trees & Ensembles](#43-regression-trees--ensembles)
5. [Model Evaluation & Metrics](#5-model-evaluation--metrics)
   - 5.1 [Classification Metrics](#51-classification-metrics)
   - 5.2 [Cross-Validation](#52-cross-validation)
   - 5.3 [ROC Curve & AUC (Binary Classification)](#53-roc-curve--auc-binary-classification)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
   - 6.1 [Grid Search CV](#61-grid-search-cv)
   - 6.2 [Randomised Search CV](#62-randomised-search-cv)
   - 6.3 [Nested Cross-Validation](#63-nested-cross-validation)
7. [Pipelines](#7-pipelines)
   - 7.1 [Building a Pipeline](#71-building-a-pipeline)
   - 7.2 [Grid Search Over Pipelines](#72-grid-search-over-pipelines)
   - 7.3 [ColumnTransformer for Mixed Data](#73-columntransformer-for-mixed-data)
8. [Lab Exercises](#8-lab-exercises)
   - [Exercise 1 – Data Loading and Exploration (Beginner)](#exercise-1--data-loading-and-exploration-beginner)
   - [Exercise 2 – Classification Comparison (Intermediate)](#exercise-2--classification-comparison-intermediate)
   - [Exercise 3 – Regression Pipeline (Intermediate)](#exercise-3--regression-pipeline-intermediate)
   - [Exercise 4 – Hyperparameter Tuning (Advanced)](#exercise-4--hyperparameter-tuning-advanced)
   - [Exercise 5 – Bonus Challenge (Expert)](#exercise-5--bonus-challenge-expert)
9. [Quick Reference](#9-quick-reference)
   - 9.1 [Algorithm Selection Guide](#91-algorithm-selection-guide)
   - 9.2 [Common Import Cheat Sheet](#92-common-import-cheat-sheet)
   - 9.3 [Evaluation Metrics Summary](#93-evaluation-metrics-summary)
   - 9.4 [Troubleshooting Common Errors](#94-troubleshooting-common-errors)
10. [Summary & Learning Outcomes](#10-summary--learning-outcomes)

---

## 1. Introduction to Scikit-learn

Scikit-learn (sklearn) is an open-source Python library built on NumPy, SciPy, and Matplotlib. It provides simple and efficient tools for data mining, data analysis, and machine learning. It is one of the most widely used ML libraries in industry and academia.

> **Objective:** By the end of this lab, students will be able to load and prepare datasets, apply classification and regression algorithms, evaluate model performance, and tune hyperparameters using Scikit-learn.

### 1.1 Why Scikit-learn?

- Consistent API across all estimators (`fit` / `predict` / `transform`)
- Rich collection of algorithms: classification, regression, clustering, dimensionality reduction
- Built-in tools for model selection, evaluation, and preprocessing
- Excellent documentation and active community
- Integrates seamlessly with NumPy, Pandas, and Matplotlib

### 1.2 Installation & Setup

Install Scikit-learn using pip or conda:

```bash
# Using pip
pip install scikit-learn

# Using conda
conda install scikit-learn
```

```python
# Verify installation
import sklearn
print(sklearn.__version__)   # e.g., 1.4.2
```

### 1.3 The Scikit-learn API Design

All Scikit-learn estimators follow a unified interface pattern:

| **Method** | **Purpose** | **Example** |
|---|---|---|
| `fit(X, y)` | Train the model on data | `model.fit(X_train, y_train)` |
| `predict(X)` | Generate predictions | `model.predict(X_test)` |
| `score(X, y)` | Evaluate model accuracy | `model.score(X_test, y_test)` |
| `transform(X)` | Transform data (preprocessors) | `scaler.transform(X_test)` |
| `fit_transform(X, y)` | Fit and transform in one step | `scaler.fit_transform(X_train)` |
| `get_params()` | Return estimator parameters | `model.get_params()` |

---

## 2. Datasets & Data Preparation

Scikit-learn provides several built-in datasets for practice and comes with utilities to generate synthetic data. This section covers loading, exploring, and splitting datasets.

### 2.1 Built-in Datasets

| **Dataset** | **Task** | **Samples** | **Features** | **Description** |
|---|---|---|---|---|
| `load_iris()` | Classification | 150 | 4 | Iris flower species (3 classes) |
| `load_breast_cancer()` | Binary Classification | 569 | 30 | Malignant vs. benign tumors |
| `load_digits()` | Multi-class | 1797 | 64 | Handwritten digit images (0–9) |
| `load_boston()` *(deprecated)* | Regression | 506 | 13 | Use `fetch_california_housing` instead |
| `fetch_california_housing()` | Regression | 20640 | 8 | California house prices |
| `make_classification()` | Classification | Custom | Custom | Synthetic classification data |
| `make_regression()` | Regression | Custom | Custom | Synthetic regression data |

### 2.2 Loading and Exploring Data

```python
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Load dataset
iris = load_iris()

# Access components
X = iris.data          # Feature matrix (150 x 4)
y = iris.target        # Target labels (0, 1, 2)
feature_names = iris.feature_names
target_names  = iris.target_names

# Create a DataFrame for exploration
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

print(df.head())
print(df.describe())
print(df['species'].value_counts())
```

### 2.3 Train / Test Split

Splitting data prevents the model from memorising training examples and ensures a fair evaluation on unseen data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% held out for testing
    random_state=42,  # Reproducibility seed
    stratify=y        # Maintain class proportions
)

print(f'Train size: {X_train.shape[0]} samples')
print(f'Test size:  {X_test.shape[0]} samples')
```

### 2.4 Feature Preprocessing

Many ML algorithms are sensitive to feature scales. Scikit-learn offers several scalers:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# StandardScaler: zero mean, unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only!
X_test_scaled  = scaler.transform(X_test)       # Apply same transform

# MinMaxScaler: scale to [0, 1] range
minmax = MinMaxScaler()
X_train_mm = minmax.fit_transform(X_train)

# Encoding categorical labels
le = LabelEncoder()
y_encoded = le.fit_transform(['cat', 'dog', 'cat', 'bird'])
print(y_encoded)  # [1 2 1 0]
```

> **Best Practice:** Always fit preprocessing transformers on training data only, then apply the fitted transformer to test data. Fitting on test data would cause data leakage.

---

## 3. Classification Algorithms

Classification is the task of predicting a discrete class label from input features. Scikit-learn provides a variety of classifiers that follow the same `fit`/`predict` interface.

### 3.1 Logistic Regression

Despite its name, Logistic Regression is a classification algorithm. It models the probability of class membership using the logistic (sigmoid) function.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Instantiate and train
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train_scaled, y_train)

# Predict
y_pred = lr.predict(X_test_scaled)

# Evaluate
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred, target_names=target_names))

# Probability estimates
proba = lr.predict_proba(X_test_scaled)
print(f'Class probabilities (first 3):\n{proba[:3]}')
```

**Key Parameters:**

| **Parameter** | **Default** | **Description** |
|---|---|---|
| `C` | 1.0 | Inverse of regularization strength; smaller = stronger regularization |
| `penalty` | `'l2'` | Regularization type: `'l1'`, `'l2'`, `'elasticnet'`, or `'none'` |
| `max_iter` | 100 | Maximum number of iterations for the solver |
| `solver` | `'lbfgs'` | Algorithm for optimization: `'lbfgs'`, `'saga'`, `'liblinear'` |
| `multi_class` | `'auto'` | `'ovr'` (one-vs-rest) or `'multinomial'` |

### 3.2 K-Nearest Neighbours (KNN)

KNN is a non-parametric algorithm that classifies a sample based on the majority class of its K nearest training neighbours.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print(f'KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}')

# Finding the best K using cross-validation
from sklearn.model_selection import cross_val_score

k_range = range(1, 21)
scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
          X_train_scaled, y_train, cv=5).mean()
          for k in k_range]

best_k = k_range[scores.index(max(scores))]
print(f'Best K: {best_k} with CV score: {max(scores):.4f}')
```

### 3.3 Support Vector Machine (SVM)

SVM finds the hyperplane that maximises the margin between classes. The kernel trick allows it to handle non-linear boundaries.

```python
from sklearn.svm import SVC

# Linear kernel
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# RBF (Radial Basis Function) kernel – handles non-linear data
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_svm = svm_rbf.predict(X_test_scaled)
print(f'SVM (RBF) Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}')
```

### 3.4 Decision Tree

A Decision Tree splits data recursively on feature thresholds, forming a tree structure. It is interpretable but prone to overfitting without constraints.

```python
from sklearn.tree import DecisionTreeClassifier, export_text

dt = DecisionTreeClassifier(
    max_depth=4,         # Limit tree depth to avoid overfitting
    min_samples_split=5, # Min samples to split a node
    min_samples_leaf=2,  # Min samples in a leaf
    random_state=42
)

dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(f'Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}')

# Visualise the tree as text
print(export_text(dt, feature_names=list(feature_names)))

# Feature importance
for name, imp in zip(feature_names, dt.feature_importances_):
    print(f'  {name}: {imp:.4f}')
```

### 3.5 Random Forest

Random Forest is an ensemble of Decision Trees trained on bootstrapped subsets of data. It reduces overfitting through averaging.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,   # Number of trees
    max_depth=None,     # Trees grow until pure leaves
    max_features='sqrt',# Features to consider at each split
    random_state=42,
    n_jobs=-1           # Use all CPU cores
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}')

# Top feature importances
importances = pd.Series(rf.feature_importances_, index=feature_names)
print(importances.sort_values(ascending=False))
```

### 3.6 Gradient Boosting

Gradient Boosting builds trees sequentially where each tree corrects the errors of the previous one. It often achieves state-of-the-art performance.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,  # Shrinkage factor
    max_depth=3,
    subsample=0.8,      # Fraction of samples per tree (stochastic GB)
    random_state=42
)

gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print(f'Gradient Boosting Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}')
```

> **Tip:** For large datasets, consider `HistGradientBoostingClassifier` — it is significantly faster than `GradientBoostingClassifier` and supports missing values natively.

---

## 4. Regression Algorithms

Regression predicts a continuous numeric output. The same Scikit-learn API pattern applies: `fit`, `predict`, and `score` using regression-specific metrics.

### 4.1 Setup for Regression

```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load regression dataset
housing = fetch_california_housing()
X_reg, y_reg = housing.data, housing.target

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_r = StandardScaler()
X_train_rs = scaler_r.fit_transform(X_train_r)
X_test_rs  = scaler_r.transform(X_test_r)

def regression_report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f'{name}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}')
```

### 4.2 Linear Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Ordinary Least Squares
ols = LinearRegression()
ols.fit(X_train_rs, y_train_r)
regression_report('Linear Regression', y_test_r, ols.predict(X_test_rs))

# Ridge Regression (L2 regularisation)
ridge = Ridge(alpha=1.0)  # alpha controls regularisation strength
ridge.fit(X_train_rs, y_train_r)
regression_report('Ridge', y_test_r, ridge.predict(X_test_rs))

# Lasso (L1 regularisation – performs feature selection)
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_rs, y_train_r)
regression_report('Lasso', y_test_r, lasso.predict(X_test_rs))

# ElasticNet (L1 + L2 combination)
en = ElasticNet(alpha=0.01, l1_ratio=0.5)
en.fit(X_train_rs, y_train_r)
regression_report('ElasticNet', y_test_r, en.predict(X_test_rs))
```

### 4.3 Regression Trees & Ensembles

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Decision Tree Regressor
dtr = DecisionTreeRegressor(max_depth=5, random_state=42)
dtr.fit(X_train_r, y_train_r)
regression_report('Decision Tree', y_test_r, dtr.predict(X_test_r))

# Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr.fit(X_train_r, y_train_r)
regression_report('Random Forest', y_test_r, rfr.predict(X_test_r))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                               max_depth=3, random_state=42)
gbr.fit(X_train_r, y_train_r)
regression_report('Gradient Boosting', y_test_r, gbr.predict(X_test_r))
```

**Regression Metrics Reference:**

| **Metric** | **Formula** | **Best Value** | **Interpretation** |
|---|---|---|---|
| MSE | `mean((y_true - y_pred)²)` | 0 | Mean squared error; penalises large errors heavily |
| RMSE | `sqrt(MSE)` | 0 | Same unit as target; easier to interpret |
| MAE | `mean(|y_true - y_pred|)` | 0 | Mean absolute error; more robust to outliers |
| R² | `1 - SS_res/SS_tot` | 1.0 | Proportion of variance explained; 1 is perfect |

---

## 5. Model Evaluation & Metrics

Evaluating model performance correctly is as important as building the model. This section covers classification and regression metrics, confusion matrices, cross-validation, and ROC curves.

### 5.1 Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Per-class and overall metrics
print('Accuracy :', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred, average='weighted'))
print('Recall   :', recall_score(y_test, y_pred, average='weighted'))
print('F1-Score :', f1_score(y_test, y_pred, average='weighted'))

# Full report
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=target_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
```

### 5.2 Cross-Validation

K-fold cross-validation gives a more reliable performance estimate by training/testing on K different splits of data.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=cv, scoring='accuracy'
)

print(f'CV Scores: {scores}')
print(f'Mean: {scores.mean():.4f} ± {scores.std():.4f}')

# Multiple metrics at once
from sklearn.model_selection import cross_validate

results = cross_validate(
    rf, X, y, cv=cv,
    scoring=['accuracy', 'f1_weighted', 'precision_weighted']
)

for metric, vals in results.items():
    if metric.startswith('test_'):
        print(f'{metric}: {vals.mean():.4f}')
```

### 5.3 ROC Curve & AUC (Binary Classification)

```python
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.datasets import load_breast_cancer

# Load binary classification dataset
bc = load_breast_cancer()
X_bc, y_bc = bc.data, bc.target

Xtr, Xte, ytr, yte = train_test_split(X_bc, y_bc, test_size=0.2,
                                        random_state=42, stratify=y_bc)

sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xte = sc.transform(Xte)

# Train with probability output
lr_bc = LogisticRegression(random_state=42).fit(Xtr, ytr)
y_prob = lr_bc.predict_proba(Xte)[:, 1]

# Compute ROC
fpr, tpr, _ = roc_curve(yte, y_prob)
roc_auc = auc(fpr, tpr)
print(f'AUC = {roc_auc:.4f}')

# Plot
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
               estimator_name='Logistic Regression').plot()
plt.show()
```

---

## 6. Hyperparameter Tuning

Hyperparameters control the learning process and are not learned from data. Scikit-learn provides Grid Search and Randomised Search to find the best hyperparameter configuration.

### 6.1 Grid Search CV

Grid Search exhaustively tries all combinations of the specified parameter grid.

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth'   : [None, 5, 10],
    'max_features': ['sqrt', 'log2'],
}

gs = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

gs.fit(X_train, y_train)
print(f'Best Parameters : {gs.best_params_}')
print(f'Best CV Score   : {gs.best_score_:.4f}')
print(f'Test Accuracy   : {gs.score(X_test, y_test):.4f}')
```

### 6.2 Randomised Search CV

Randomised Search samples parameter combinations randomly — much faster than Grid Search for large spaces.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth'   : [None, 3, 5, 10, 20],
    'max_features': ['sqrt', 'log2', 0.5],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf' : randint(1, 10),
}

rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,        # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

rs.fit(X_train, y_train)
print(f'Best Params: {rs.best_params_}')
print(f'Test Accuracy: {rs.score(X_test, y_test):.4f}')
```

### 6.3 Nested Cross-Validation

Nested CV separates model selection from model evaluation, giving an unbiased estimate of generalisation performance.

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

gs_inner = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {'n_estimators': [50, 100], 'max_depth': [None, 5]},
    cv=inner_cv, scoring='accuracy'
)

nested_scores = cross_val_score(gs_inner, X, y, cv=outer_cv, scoring='accuracy')
print(f'Nested CV: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}')
```

---

## 7. Pipelines

Pipelines chain preprocessing steps and a final estimator into a single object. This prevents data leakage, simplifies code, and makes deployment safer.

### 7.1 Building a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Each step is a (name, estimator) tuple
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', probability=True, random_state=42))
])

# Fit and predict just like a regular estimator
pipe.fit(X_train, y_train)
y_pipe = pipe.predict(X_test)
print(f'Pipeline Accuracy: {accuracy_score(y_test, y_pipe):.4f}')

# Access individual steps
print(pipe.named_steps['scaler'].mean_)
print(pipe.named_steps['svm'].support_vectors_.shape)
```

### 7.2 Grid Search Over Pipelines

Use double underscore notation (`step__param`) to tune pipeline parameters:

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA()),
    ('clf',    RandomForestClassifier(random_state=42))
])

param_grid = {
    'pca__n_components' : [2, 4, None],
    'clf__n_estimators' : [50, 100],
    'clf__max_depth'    : [None, 5],
}

gs_pipe = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs_pipe.fit(X_train, y_train)
print(f'Best Pipeline Score: {gs_pipe.best_score_:.4f}')
print(f'Best Params: {gs_pipe.best_params_}')
```

### 7.3 ColumnTransformer for Mixed Data

Real-world datasets often have both numerical and categorical features requiring different preprocessing.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

num_features = ['age', 'income', 'credit_score']
cat_features = ['gender', 'occupation', 'region']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

full_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   RandomForestClassifier(n_estimators=100, random_state=42))
])
```

---

## 8. Lab Exercises

> Complete the following exercises in order. Each builds on concepts introduced in previous sections. Submit your Jupyter Notebook file with all cells executed.

### Exercise 1 – Data Loading and Exploration (Beginner)

- Load the breast cancer dataset using `load_breast_cancer()`.
- Create a Pandas DataFrame with feature names as columns.
- Print the shape, summary statistics, and class distribution.
- Split into train/test sets (80/20, stratified, `random_state=0`).
- Apply `StandardScaler`; confirm the training mean is approximately zero.

### Exercise 2 – Classification Comparison (Intermediate)

- Using the breast cancer dataset from Exercise 1, train the following classifiers: Logistic Regression, KNN (`k=7`), SVM (RBF kernel), Random Forest (100 trees).
- For each classifier, report: Accuracy, Precision (macro), Recall (macro), F1-score (macro).
- Create a bar chart comparing the four classifiers by accuracy.
- Plot the ROC curve for all four classifiers on the same axes and report AUC scores.
- Which classifier performs best? Justify your answer.

### Exercise 3 – Regression Pipeline (Intermediate)

- Load `fetch_california_housing()` and inspect the dataset.
- Build a Pipeline with `StandardScaler` → `LinearRegression`.
- Evaluate using 5-fold cross-validation; report mean RMSE and R².
- Repeat with `GradientBoostingRegressor(n_estimators=200)` in the pipeline.
- Which model generalises better and why?

### Exercise 4 – Hyperparameter Tuning (Advanced)

- Load the digits dataset (`load_digits()`).
- Build a Pipeline: `StandardScaler` → `PCA` → `SVC`.
- Use `GridSearchCV` (`cv=5`) to tune:
  - `pca__n_components` in `[10, 20, 30, 40]`
  - `svc__C` in `[0.1, 1, 10, 100]`
  - `svc__kernel` in `['linear', 'rbf']`
- Report the best parameters and their cross-validated accuracy.
- Evaluate the best estimator on a held-out test set (20%). Report the confusion matrix.

### Exercise 5 – Bonus Challenge (Expert)

- Find or create a dataset with both numerical and categorical features (e.g., the Titanic dataset from an open source).
- Build a full `ColumnTransformer` pipeline handling missing values, encoding, and scaling.
- Perform nested cross-validation to get an unbiased performance estimate.
- Use `RandomizedSearchCV` (`n_iter=30`) to tune hyperparameters.
- Plot learning curves to diagnose bias vs. variance.

---

## 9. Quick Reference

### 9.1 Algorithm Selection Guide

| **Problem Type** | **Small Dataset (< 10K)** | **Large Dataset (> 10K)** |
|---|---|---|
| Binary Classification | SVM, Logistic Regression, KNN | Gradient Boosting, Random Forest, SGD Classifier |
| Multi-class Classification | Decision Tree, Random Forest, SVM (OvR) | Gradient Boosting, HistGBT, Neural Networks |
| Regression | Ridge, Lasso, ElasticNet, SVR | Gradient Boosting, Random Forest Regressor |
| Clustering | KMeans, DBSCAN, AgglomerativeClustering | MiniBatchKMeans, BIRCH |
| Dimensionality Reduction | PCA, LDA, t-SNE | PCA, TruncatedSVD, UMAP |

### 9.2 Common Import Cheat Sheet

```python
# Data loading & splitting
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
```

### 9.3 Evaluation Metrics Summary

| **Metric** | **Use For** | **sklearn Function** | **Notes** |
|---|---|---|---|
| Accuracy | Classification | `accuracy_score(y_true, y_pred)` | Misleading for imbalanced classes |
| Precision | Classification | `precision_score(..., average='weighted')` | TP / (TP + FP) |
| Recall | Classification | `recall_score(..., average='weighted')` | TP / (TP + FN) |
| F1-Score | Classification | `f1_score(..., average='weighted')` | Harmonic mean of P & R |
| ROC-AUC | Binary Classif. | `roc_auc_score(y_true, y_prob)` | Probability of correct ranking |
| RMSE | Regression | `np.sqrt(mean_squared_error(...))` | Same unit as target |
| MAE | Regression | `mean_absolute_error(y_true, y_pred)` | More robust to outliers |
| R² | Regression | `r2_score(y_true, y_pred)` | 1.0 is perfect; can be negative |

### 9.4 Troubleshooting Common Errors

| **Error / Issue** | **Likely Cause** | **Fix** |
|---|---|---|
| `ValueError: Input contains NaN` | Missing values in features | Use `SimpleImputer` before model |
| `ConvergenceWarning` in LogReg | `max_iter` too low | Increase `max_iter` or scale features |
| `DataConversionWarning` | `y` is column vector, not 1-D | Use `ravel()` or `reshape(-1)` |
| Memory error with SVM | Dataset too large for `SVC` | Use `LinearSVC` or `SGDClassifier` |
| Overfitting (train >> test) | Model too complex | Add regularisation or reduce depth |
| Data leakage in CV | Scaler fit on all data | Use `Pipeline` to fit inside CV loop |

---

## 10. Summary & Learning Outcomes

After completing this lab, you should be able to:

- Load, explore, and split datasets using Scikit-learn's dataset utilities and `train_test_split`
- Apply `StandardScaler` and `MinMaxScaler` correctly, fitting only on training data
- Train and compare classification algorithms: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting
- Train and compare regression algorithms: Linear Regression, Ridge, Lasso, Tree-based regressors
- Evaluate models using appropriate metrics: accuracy, precision, recall, F1, AUC for classification; RMSE, MAE, R² for regression
- Use cross-validation and nested cross-validation for unbiased performance estimation
- Tune hyperparameters with `GridSearchCV` and `RandomizedSearchCV`
- Build `Pipeline`s combining preprocessing and estimators to prevent data leakage
- Handle mixed-type datasets with `ColumnTransformer`

### Further Reading & Resources

- Official Documentation: https://scikit-learn.org/stable/user_guide.html
- API Reference: https://scikit-learn.org/stable/modules/classes.html
- Scikit-learn Tutorials: https://scikit-learn.org/stable/tutorial/index.html
- Book: *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* – Aurélien Géron
- Book: *Introduction to Machine Learning with Python* – Müller & Guido

---

> **Lab Complete!** Submit your Jupyter Notebook (`.ipynb`) with all exercises completed and cells executed. Ensure your code is well-commented and outputs are visible.
