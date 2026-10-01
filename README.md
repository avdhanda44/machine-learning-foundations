# Machine Learning

This repository is organized as a complete machine learning learning path.

Each topic has two kinds of notebooks:

- `_Notes.ipynb` files explain the theory, mechanism, formulas, terms, intuition, strengths, weaknesses, and evaluation.
- Matching code notebooks show practical implementation separately, so the theory notes stay clean and continuous.

## Course Order

| No. | Topic | Notes | Code |
| --- | --- | --- | --- |
| 00 | Algorithms Introduction | [00_Algorithms_Introduction.ipynb](00_Algorithms_Introduction.ipynb) | - |
| 01 | Automated EDA | [01_Automated_EDA_Notes.ipynb](01_Automated_EDA_Notes.ipynb) | [01_Automated_EDA.ipynb](01_Automated_EDA.ipynb) |
| 02 | Linear Regression | [02_Linear_Regression_Notes.ipynb](02_Linear_Regression_Notes.ipynb) | [02_Linear_Regression.ipynb](02_Linear_Regression.ipynb) |
| 03 | Ridge Regression | [03_Ridge_Regression_Notes.ipynb](03_Ridge_Regression_Notes.ipynb) | [03_Ridge_Regression.ipynb](03_Ridge_Regression.ipynb) |
| 04 | Lasso Regression | [04_Lasso_Regression_Notes.ipynb](04_Lasso_Regression_Notes.ipynb) | [04_Lasso_Regression.ipynb](04_Lasso_Regression.ipynb) |
| 05 | Elastic Net Regression | [05_Elastic_Net_Regression_Notes.ipynb](05_Elastic_Net_Regression_Notes.ipynb) | [05_Elastic_Net_Regression.ipynb](05_Elastic_Net_Regression.ipynb) |
| 06 | Polynomial Regression | [06_Polynomial_Regression_Notes.ipynb](06_Polynomial_Regression_Notes.ipynb) | [06_Polynomial_Regression.ipynb](06_Polynomial_Regression.ipynb) |
| 07 | Logistic Regression | [07_Logistic_Regression_Notes.ipynb](07_Logistic_Regression_Notes.ipynb) | [07_Logistic_Regression.ipynb](07_Logistic_Regression.ipynb) |
| 08 | KNN Algorithm | [08_KNN_Algorithm_Notes.ipynb](08_KNN_Algorithm_Notes.ipynb) | [08_KNN_Algorithm.ipynb](08_KNN_Algorithm.ipynb) |
| 09 | Naive Bayes | [09_Naive_Bayes_Notes.ipynb](09_Naive_Bayes_Notes.ipynb) | [09_Naive_Bayes.ipynb](09_Naive_Bayes.ipynb) |
| 10 | Support Vector Machine | [10_SVM_Notes.ipynb](10_SVM_Notes.ipynb) | [10_SVM.ipynb](10_SVM.ipynb) |
| 11 | Decision Tree | [11_Decision_Tree_Notes.ipynb](11_Decision_Tree_Notes.ipynb) | [11_Decision_Tree.ipynb](11_Decision_Tree.ipynb) |
| 12 | Random Forest | [12_Random_Forest_Notes.ipynb](12_Random_Forest_Notes.ipynb) | [12_Random_Forest.ipynb](12_Random_Forest.ipynb) |
| 13 | Ensemble Learning Techniques | [13_Ensemble_Learning_Techniques_Notes.ipynb](13_Ensemble_Learning_Techniques_Notes.ipynb) | [13_Ensemble_Learning_Techniques.ipynb](13_Ensemble_Learning_Techniques.ipynb) |
| 14 | AdaBoost | [14_AdaBoost_Notes.ipynb](14_AdaBoost_Notes.ipynb) | [14_AdaBoost.ipynb](14_AdaBoost.ipynb) |
| 15 | Gradient Boosting | [15_Gradient_Boosting_Notes.ipynb](15_Gradient_Boosting_Notes.ipynb) | [15_Gradient_Boosting.ipynb](15_Gradient_Boosting.ipynb) |
| 16 | XGBoost | [16_XGBoost_Notes.ipynb](16_XGBoost_Notes.ipynb) | [16_XGBoost.ipynb](16_XGBoost.ipynb) |
| 17 | LightGBM | [17_LightGBM_Notes.ipynb](17_LightGBM_Notes.ipynb) | [17_LightGBM.ipynb](17_LightGBM.ipynb) |
| 18 | CatBoost | [18_CatBoost_Notes.ipynb](18_CatBoost_Notes.ipynb) | [18_CatBoost.ipynb](18_CatBoost.ipynb) |
| 19 | K-Means | [19_KMeans_Notes.ipynb](19_KMeans_Notes.ipynb) | [19_KMeans.ipynb](19_KMeans.ipynb) |
| 20 | Silhouette Coefficient | [20_Silhouette_Coefficient_Notes.ipynb](20_Silhouette_Coefficient_Notes.ipynb) | [20_Silhouette_Coefficient.ipynb](20_Silhouette_Coefficient.ipynb) |
| 21 | DBSCAN | [21_DBSCAN_Notes.ipynb](21_DBSCAN_Notes.ipynb) | [21_DBSCAN.ipynb](21_DBSCAN.ipynb) |
| 22 | Hierarchical Clustering | [22_Hierarchical_Clustering_Notes.ipynb](22_Hierarchical_Clustering_Notes.ipynb) | [22_Hierarchical_Clustering.ipynb](22_Hierarchical_Clustering.ipynb) |
| 23 | Gaussian Mixture Models | [23_Gaussian_Mixture_Models_Notes.ipynb](23_Gaussian_Mixture_Models_Notes.ipynb) | [23_Gaussian_Mixture_Models.ipynb](23_Gaussian_Mixture_Models.ipynb) |
| 24 | Principal Component Analysis | [24_PCA_Notes.ipynb](24_PCA_Notes.ipynb) | [24_PCA.ipynb](24_PCA.ipynb) |

## Practical Workflow

[25_End_to_End_Workflow.ipynb](25_End_to_End_Workflow.ipynb) combines missing-value
imputation, categorical encoding, `ColumnTransformer`, fold-local preprocessing,
a baseline, cross-validation, hyperparameter search, and final held-out evaluation.
The data is generated locally and requires no download.

## Setup and Checks

Use Python 3.11 (the CI version):

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

On macOS, LightGBM may require the OpenMP runtime (`brew install libomp`).

```sh
python scripts/check_notebooks.py --validate-only
python scripts/check_notebooks.py
```

The full check validates every notebook and executes notebooks containing code in
fresh kernels and temporary working directories. All three external boosting
libraries are required so their examples cannot silently skip execution. Failures
produce a nonzero exit status. Executed notebooks, including outputs, are saved
under `artifacts/executed/`; GitHub Actions runs the same check and uploads these
as downloadable artifacts. Source notebooks stay output-free. Execution checks
detect runtime failures, not every statistical or conceptual mistake.

The Ridge, Lasso, Elastic Net, and KNN companions select hyperparameters using
training-fold cross-validation, then evaluate the selected model on held-out data.

## Learning Style

The notes are written for understanding from first principles. They avoid short revision-note formatting and keep code in separate notebooks. The code notebooks are meant to support the theory with clean, runnable examples.
