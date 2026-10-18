# Brain Tumor Classification using MRI ADC Texture Features

An end-to-end Machine Learning pipeline developed in Python to classify brain tumor pathology from Apparent Diffusion Coefficient (ADC) MRI texture features. 

This project explores feature extraction, data preprocessing, and ensemble tree-based models to achieve high-precision classification in medical imaging datasets.

---

## 📌 Project Overview

- **Domain:** Medical Imaging / Machine Learning
- **Task:** Multiclass / Binary Tumor Classification based on MRI texture descriptors
- **Key Challenges:** High dimensionality, feature correlation, and class imbalance in clinical datasets
- **Primary Metrics:** Precision, Recall, Confusion Matrix, and F1-Score (optimized via cross-validation)

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python
- **Data Processing & Analysis:** NumPy, Pandas
- **Machine Learning & Modeling:** Scikit-Learn (Random Forest, Extra Trees, Decision Trees)
- **Validation:** Stratified 5-Fold Cross-Validation
- **Visualization:** Matplotlib, Seaborn

---

## 🔬 Methodology & Architecture

1. **Data Preprocessing & Cleaning:**
   - Handling missing values and feature scaling/normalization.
   - Exploratory Data Analysis (EDA) on ADC texture parameters.
2. **Feature Engineering:**
   - Evaluation of texture metrics and dimensionality analysis.
3. **Model Selection & Training:**
   - Benchmarking tree-based ensembles: **Random Forest** and **ExtraTreesClassifier**.
   - Hyperparameter tuning to avoid overfitting on limited medical samples.
4. **Validation:**
   - 5-Fold Cross-Validation to guarantee model generalizability and prevent data leakage.
   - Comprehensive performance evaluation focusing on weighted **F1-Score**.

