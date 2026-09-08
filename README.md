```markdown
# Machine Learning Assignments

This repository contains completed machine learning assignments and their supporting datasets.

## Assignment 3: Decision Trees and Random Forests

[Open the completed notebook](Decision_trees_and_random_trees.ipynb)

The notebook uses the UCI Phishing Websites dataset to predict whether a website is phishing or legitimate.

### Contents

- Part 1: A small, interpretable decision tree
- Decision tree visualization and manual classification instructions
- Part 2: A random forest classifier
- Hyperparameter tuning with five-fold cross-validation
- Comparison of decision tree and random forest performance
- Confusion matrix and classification metrics
- Permutation feature importance
- Conclusions and deployment considerations

### Random forest results

- Test accuracy: 96.4%
- Phishing precision: 96.2%
- Phishing recall: 97.4%
- Phishing F1 score: 96.8%
- ROC AUC: 99.2%

The tuned random forest reduced the number of missed phishing websites from 30 to 9 compared with the depth-2 decision tree.

### Files

- `Decision_trees_and_random_trees.ipynb` — completed assignment notebook
- `data/phishing_websites.csv` — dataset used by the notebook
- `data/phishing+websites/` — original dataset files and documentation

## Data Preprocessing Assignment

[Open the completed notebook](CKD_data_preprocessing_assignment.ipynb)

This notebook uses the UCI Chronic Kidney Disease dataset. It includes data cleaning, descriptive statistics, histograms, outlier analysis, correlation heatmaps, and written interpretation.

### Files

- `CKD_data_preprocessing_assignment.ipynb` — completed assignment notebook
- `data/chronic_kidney_disease.csv` — local copy of the UCI dataset
- `uci_metadata.json` — UCI dataset metadata
- `build_notebook.py` — reproducible notebook generator

## Running the notebooks

Open a notebook in Jupyter Notebook or JupyterLab and run all cells using a Python 3 environment containing:

- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- IPython

## Data sources

- Phishing Websites, UCI Machine Learning Repository:  
  https://archive.ics.uci.edu/dataset/327/phishing+websites
- Chronic Kidney Disease, UCI Machine Learning Repository:  
  https://doi.org/10.24432/C5G020
```
