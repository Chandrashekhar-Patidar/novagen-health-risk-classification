# Novagen Labs – Health Risk Classification Using Machine Learning

A comprehensive Machine Learning classification project that implements and compares **12 machine learning model families** using health and lifestyle data. The project explores traditional classification algorithms, hyperparameter optimization, decision tree pruning, and advanced ensemble learning techniques to identify the best-performing model.

**Best Model:** Stacking Classifier  
**Test Accuracy:** 95.11%  
**Test Recall:** 95.56%

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Project Workflow](#3-project-workflow)
4. [Machine Learning Models](#4-machine-learning-models)
5. [Model Performance and Comparison](#5-model-performance-and-comparison)
6. [Key Findings and Insights](#6-key-findings-and-insights)
7. [Repository Structure](#7-repository-structure)
8. [Installation and Execution](#8-installation-and-execution)
9. [Technologies and Libraries](#9-technologies-and-libraries)
10. [Limitations and Future Improvements](#10-limitations-and-future-improvements)
11. [Disclaimer](#11-disclaimer)
12. [Author](#12-author)

---

## 1. Project Overview

The **Novagen Labs Health Risk Classification** project focuses on developing and evaluating supervised Machine Learning classification models using a structured dataset containing health measurements, lifestyle habits, and personal attributes.

The objective is to predict a binary target variable (`Target`: 0 or 1) based on the available input features and compare different classification techniques to identify the model with the strongest predictive performance.

Rather than relying on a single algorithm, this project explores multiple approaches, starting with traditional classification methods and progressing towards ensemble learning techniques.

### Key Highlights

- Implementation of 12 machine learning model families.
- Comparison of 14 model configurations, including different Decision Tree variants.
- Data preprocessing and numerical feature scaling.
- Hyperparameter optimization using GridSearchCV.
- Decision Tree Pre-Pruning and Post-Pruning.
- Implementation of Bagging and Boosting techniques.
- Heterogeneous ensemble learning using Voting and Stacking Classifiers.
- Model evaluation using Accuracy, Recall, Precision, F1-Score, and Classification Reports.
- Comparative analysis to identify the best-performing model.

---

## 2. Dataset Description

The project uses a structured health and lifestyle dataset containing numerical and categorical attributes.

| Dataset Information | Details |
|---|---|
| Dataset File | `novagen_dataset.csv` |
| Total Records | 9,549 |
| Total Columns | 23 |
| Input Features | 22 |
| Target Variable | `Target` |
| Problem Type | Binary Classification |
| Missing Values | 0 |

### Feature Description

The dataset contains the following groups of features:

| Feature Category | Features |
|---|---|
| Health Measurements | `Age`, `BMI`, `Blood_Pressure`, `Cholesterol`, `Glucose_Level`, `Heart_Rate` |
| Lifestyle Measurements | `Sleep_Hours`, `Exercise_Hours`, `Water_Intake`, `Stress_Level` |
| Lifestyle and Background Attributes | `Smoking`, `Alcohol`, `Diet`, `MentalHealth`, `PhysicalActivity`, `MedicalHistory`, `Allergies` |
| One-Hot Encoded Features | `Diet_Type__Vegan`, `Diet_Type__Vegetarian`, `Blood_Group_AB`, `Blood_Group_B`, `Blood_Group_O` |

### Target Variable

The dataset contains a binary target variable:

- `0` – Class 0
- `1` – Class 1

The dataset does not provide a documented interpretation of these two labels, so they are treated as classification classes rather than specific medical conditions.

### Train-Test Distribution

The dataset is divided into training and testing subsets using a 70:30 split.

| Dataset | Records |
|---|---:|
| Training Set | 6,684 |
| Testing Set | 2,865 |
| Total | 9,549 |

The test set contains 1,356 samples from Class 0 and 1,509 samples from Class 1.

---

## 3. Project Workflow

The project follows a structured Machine Learning workflow, from data preparation to model evaluation.

### Step 1: Data Loading

- Imported the dataset using Pandas.
- Examined the dataset structure, data types, and sample records.
- Identified numerical, categorical, and Boolean features.

### Step 2: Data Preprocessing

- Converted Boolean features into numerical representations.
- Mapped `True` and `False` values to `1` and `0`.
- Converted Boolean blood group indicators into integer values.
- Separated independent variables (`X`) and the target variable (`y`).

### Step 3: Train-Test Splitting

Used `train_test_split()` to divide the dataset:

- Training data: 70%
- Testing data: 30%
- Random state: `42`

The same held-out test set is used for comparing the evaluated models.

### Step 4: Feature Scaling

Applied `StandardScaler` to the following 10 numerical features:

- Age
- BMI
- Blood Pressure
- Cholesterol
- Glucose Level
- Heart Rate
- Sleep Hours
- Exercise Hours
- Water Intake
- Stress Level

The scaler is fitted only on the training data and subsequently applied to the test data to avoid using test-set statistics during scaling.

### Step 5: Model Training and Hyperparameter Tuning

Implemented multiple classification algorithms and used `GridSearchCV` with cross-validation for selected models to identify suitable hyperparameters.

### Step 6: Model Evaluation

Evaluated model performance using:

- Accuracy Score
- Recall Score
- Precision
- F1-Score
- Classification Report
- Cross-Validation Score
- Out-of-Bag Score (Random Forest)

### Step 7: Model Comparison

Compared the performance of all evaluated model configurations to identify the strongest classification approach.

---

## 4. Machine Learning Models

This project explores 12 different model families, covering individual classification algorithms and ensemble learning techniques.

### 4.1 Individual Classification Algorithms

**1. Logistic Regression**

Implemented as a baseline classification algorithm to establish an initial performance benchmark.

**2. K-Nearest Neighbors (KNN)**

Implemented KNN and optimized the number of neighbors using GridSearchCV.

- Parameters tested: `n_neighbors = [3, 5, 7, 9]`
- Cross-validation: 5-fold
- Optimization metric: Recall

**3. Support Vector Machine (SVM)**

Implemented SVM with multiple kernel functions and optimized its hyperparameters.

- Kernels: Linear, Polynomial, RBF, Sigmoid
- C values: `[0.1, 1, 10, 100]`
- Cross-validation: 3-fold
- Optimization metric: Accuracy
- Best parameters: `kernel = 'rbf'`, `C = 10`

**4. Gaussian Naive Bayes**

Implemented Gaussian Naive Bayes as a probabilistic classification algorithm based on the assumption of Gaussian-distributed features.

**5. Decision Tree Classifier**

Explored three Decision Tree configurations:

- Default Decision Tree: Implemented using default hyperparameters.
- Pre-Pruning: Optimized `max_depth`, `min_samples_split`, and `min_samples_leaf` using GridSearchCV.
- Post-Pruning: Applied Cost Complexity Pruning using `ccp_alpha` and cross-validation.

Also visualized the optimized pruned Decision Tree using Matplotlib.

### 4.2 Ensemble Learning Algorithms

**6. Random Forest**

Implemented Random Forest using multiple Decision Trees trained on bootstrap samples.

- Number of estimators: 201
- Out-of-Bag (OOB) evaluation enabled.
- OOB score: 93.81%

**7. Bagging Classifier**

Implemented Bagging using multiple Decision Tree estimators trained on different bootstrap samples, with predictions aggregated to improve stability.

- Base estimator: Decision Tree
- Number of estimators: 100

**8. Gradient Boosting**

Implemented Gradient Boosting, where sequential models improve predictions by learning from the errors of previous models.

- Number of estimators: 200
- Learning rate: 0.1
- Maximum depth: 5

**9. AdaBoost**

Implemented Adaptive Boosting, which gives greater importance to previously misclassified training samples.

- Base estimator: Decision Tree with maximum depth 5
- Number of estimators: 200

**10. XGBoost**

Implemented XGBoost, an optimized gradient boosting algorithm that incorporates regularization and computational improvements.

- Number of estimators: 200
- Maximum depth: 5
- Learning rate: 0.1
- Evaluation metric: Log Loss

### 4.3 Heterogeneous Ensemble Learning

**11. Voting Classifier**

Combined six different classification algorithms:

- Logistic Regression
- Support Vector Machine
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Random Forest

The classifier uses hard voting to determine the final prediction based on the majority of individual model predictions.

**12. Stacking Classifier**

Implemented Stacking using the same six base estimators.

Unlike Voting, Stacking uses the predictions of the base models as inputs to a final estimator, which learns how to combine their outputs.

- Base estimators: 6 different classification algorithms
- Cross-validation: 5-fold
- Final estimator: Scikit-learn's default Logistic Regression estimator

---

## 5. Model Performance and Comparison

All model configurations were evaluated on the same held-out test dataset containing 2,865 records.

The following table presents the recorded Accuracy and Recall scores.

| Rank | Model | Accuracy | Recall |
|---:|---|---:|---:|
| 1 | **Stacking Classifier** | **95.11%** | **0.956** |
| 2 | XGBoost | 94.87% | 0.952 |
| 3 | Gradient Boosting | 94.66% | 0.954 |
| 4 | AdaBoost | 94.59% | 0.950 |
| 5 | SVM (Tuned) | 94.49% | 0.951 |
| 6 | Random Forest | 94.24% | 0.952 |
| 7 | Bagging Classifier | 94.07% | 0.939 |
| 8 | Voting Classifier | 93.19% | 0.928 |
| 9 | KNN (Tuned) | 92.22% | 0.920 |
| 10 | Decision Tree (Post-Pruned) | 90.86% | 0.905 |
| 11 | Decision Tree (Default) | 90.26% | 0.907 |
| 12 | Decision Tree (Pre-Pruned) | 90.23% | 0.892 |
| 13 | Gaussian Naive Bayes | 82.44% | 0.820 |
| 14 | Logistic Regression | 81.82% | 0.831 |

*Note: There are 14 evaluated configurations across 12 model families because three Decision Tree variants are evaluated separately. Recall refers to the positive class (Class 1). Values are rounded to three decimal places where applicable.*

### Best Performing Model: Stacking Classifier

The **Stacking Classifier** achieved the best performance with **95.11% accuracy** and **95.56% recall**, combining multiple classifiers through a final estimator. XGBoost followed closely with 94.87% accuracy.

---

## 6. Key Learnings

- Data preprocessing, encoding, and feature scaling.
- Implementation of multiple classification algorithms.
- Hyperparameter tuning using `GridSearchCV`.
- Decision Tree Pre-Pruning and Post-Pruning.
- Bagging, Boosting, Voting, and Stacking techniques.
- Model evaluation and performance comparison.

## 7. Repository Structure

```text
novagen-labs-health-risk-classification/
│
├── dataset/
│   └── novagen_dataset.csv
│
├── notebooks/
│   └── Novagen_Labs_Project.ipynb
│
├── README.md
└── requirements.txt
```

## 8. Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/Chandrashekhar-Patidar/novagen-labs-health-risk-classification.git
cd novagen-labs-health-risk-classification
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Launch JupyterLab**

```bash
jupyter lab
```

Open `notebooks/Novagen_Labs_Project.ipynb` and run the cells sequentially.

## 9. Technologies Used

- **Python** – Programming language
- **Pandas & NumPy** – Data processing
- **Scikit-learn** – Model training, tuning, and evaluation
- **XGBoost** – Gradient boosting
- **Matplotlib** – Data visualization
- **JupyterLab** – Development environment

## 10. Disclaimer

This project is developed for educational purposes. Its predictions are not intended for medical diagnosis or healthcare decision-making.

## 12. Author

**Chandrashekhar Patidar**

GitHub: [Chandrashekhar-Patidar](https://github.com/Chandrashekhar-Patidar)

---
