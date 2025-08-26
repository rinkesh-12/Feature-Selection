# Feature Selection & Overfitting Control - README

This README provides a **cheat sheet** for Feature Selection, Model Performance Optimization, and Overfitting Control. It is designed to help you quickly revise for interviews and apply best practices in real-world projects.

---

## 🔑 1. Basic Feature Selection Methods

* **Remove Constant / Quasi-Constant Features**: Drop features with no variance.
* **Correlation-based Removal**: Remove highly correlated features (multicollinearity).
* **Chi-Square Test**: For categorical features vs classification target.
* **ANOVA (f-test)**: Numerical features vs categorical target.
* **Mutual Information / Information Gain**: Works for both classification & regression.

---

## 🚀 2. Advanced Feature Selection Techniques

* **Recursive Feature Elimination (RFE)**: Iteratively removes weakest features.
* **Embedded Methods**:

  * **Lasso (L1 Regularization)**: Shrinks some coefficients to zero.
  * **Ridge (L2 Regularization)**: Reduces variance but keeps all features.
  * **ElasticNet**: Combination of L1 + L2.
  * **Tree-based Feature Importance** (Random Forest, XGBoost, LightGBM).
* **Dimensionality Reduction**:

  * **PCA (Principal Component Analysis)**: Reduces dimensions while preserving variance.
  * **Autoencoder-based Feature Extraction**: In deep learning contexts.

---

## 🛡️ 3. Overfitting Control

* **Cross-Validation (K-Fold)**: Ensures generalization.
* **Regularization**: L1, L2, ElasticNet.
* **Early Stopping**: Prevents over-training (used in boosting, neural nets).
* **Dropout**: For neural networks.
* **Pruning**: For decision trees.
* **Feature Selection**: Reduces noisy inputs.

---

## 🛠️ 4. Feature Engineering (Always Needed)

* **Handling Missing Values**: Mean/Median/Mode imputation, advanced methods (KNN, MICE).
* **Encoding Categorical Variables**:

  * One-hot encoding
  * Label encoding
  * Target encoding
* **Scaling Features**:

  * StandardScaler, MinMaxScaler, RobustScaler.
* **Interaction Terms**: Polynomial & cross features.
* **Domain Features**: Based on subject knowledge.

---

## 📊 5. Feature Evaluation

* **Feature Importance (Tree Models)**: Built-in ranking.
* **Permutation Importance**: Shuffling values to test impact.
* **Model Explainability Tools**:

  * **SHAP (SHapley values)**
  * **LIME (Local Interpretable Model-agnostic Explanations)**

---

## 📖 6. Key Theoretical Knowledge for Interviews

* **Curse of Dimensionality**: More features → harder learning.
* **Bias-Variance Tradeoff**: Core of model generalization.
* **Correlation Issues**: Multicollinearity reduces interpretability.
* **Filter vs Wrapper vs Embedded Methods**:

  * Filter: Chi-Square, ANOVA, Correlation.
  * Wrapper: RFE.
  * Embedded: Lasso, Tree importance.
* **PCA vs Feature Selection**: PCA transforms, FS selects.
* **Why Regularization Helps**: Penalizes complexity, reduces overfitting.

---

## ✅ Interview-Ready Answer Template

**Question**: How do you select features in your projects?

**Answer**:

1. Remove constant and quasi-constant features.
2. Remove highly correlated features.
3. Apply univariate tests (Chi-Square, ANOVA, MI).
4. Use embedded models (Lasso, RF, XGBoost) for importance.
5. Validate selection with cross-validation.

---

This cheat sheet will help you in **interviews** and in **practical projects** for faster processing, accurate results, and reducing overfitting.

----------------------------------------------------

# Feature Selection & Overfitting Control - Interview Guide

This document provides a **comprehensive cheat sheet** for Feature Selection, Model Performance Optimization, Overfitting Control, and Interview Preparation. It includes **Q\&A with formulas** commonly asked in data science and machine learning interviews.

---

## 🔑 1. Feature Selection Basics

### Q1: What are the types of feature selection methods?

**Answer**:

* **Filter Methods**: Correlation, Chi-Square, ANOVA, Mutual Information.
* **Wrapper Methods**: Recursive Feature Elimination (RFE).
* **Embedded Methods**: Lasso, Ridge, Decision Trees, XGBoost.

---

### Q2: How do you detect and handle multicollinearity?

* Use **Correlation Matrix** (remove features with |r| > 0.8).
* Use **Variance Inflation Factor (VIF)**.

📌 **Formula**:

$$
VIF_i = \frac{1}{1 - R_i^2}
$$

Where $R_i^2$ is the R² score when regressing feature $i$ on all other features.
👉 If **VIF > 10**, drop that feature.

---

### Q3: How do you measure dependency between features and target?

Use **Mutual Information (MI)**.

📌 **Formula**:

$$
I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log \left( \frac{p(x,y)}{p(x)p(y)} \right)
$$

---

### Q4: What is PCA? When do you use it?

**Answer**: PCA reduces dimensionality by projecting features into new uncorrelated components. Used in high-dimensional or correlated data.

📌 **PCA Steps**:

1. Standardize data.
2. Compute covariance matrix:

   $$
   C = \frac{1}{n-1} (X - \bar{X})^T (X - \bar{X})
   $$
3. Compute eigenvectors & eigenvalues.
4. Select top $k$ eigenvectors as new features.

---

## 🛡️ 2. Overfitting Control

### Q5: How do you detect overfitting?

* Training accuracy >> Validation accuracy.
* High variance error.

### Q6: How do you prevent overfitting?

* Regularization (L1, L2).
* Cross-validation.
* Early stopping.
* Dropout (for neural nets).
* Tree pruning.
* Feature selection.

📌 **Regularization Formulas**:

* **Ridge (L2)**:

$$
J(\theta) = \sum (y_i - \hat{y}_i)^2 + \lambda \sum \theta_j^2
$$

* **Lasso (L1)**:

$$
J(\theta) = \sum (y_i - \hat{y}_i)^2 + \lambda \sum |\theta_j|
$$

👉 Lasso shrinks some coefficients to **zero** (feature selection).

---

## 📊 3. Model Evaluation Metrics

### Q7: How do you evaluate classification performance?

* Accuracy, Precision, Recall, F1 Score.

📌 **Formulas**:

* Precision = $\frac{TP}{TP+FP}$
* Recall = $\frac{TP}{TP+FN}$
* F1 Score = $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$

---

### Q8: How do you evaluate regression performance?

* R², Adjusted R², RMSE, MAE.

📌 **Formulas**:

* R²:

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

* Adjusted R²:

$$
R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
$$

Where $n$ = samples, $p$ = features.

👉 Adjusted R² penalizes adding useless features.

---

### Q9: Why is cross-validation better than train-test split?

**Answer**: Train-test split may give biased results. Cross-validation ensures robust and stable performance.

📌 **K-Fold CV Error Formula**:

$$
CV_{error} = \frac{1}{K} \sum_{k=1}^{K} error_k
$$

---

## 📖 4. Key Theoretical Knowledge

### Q10: What is the Curse of Dimensionality?

* As features increase, data becomes sparse → models struggle to learn.

### Q11: Explain Bias-Variance Tradeoff.

* **High bias** → underfitting.
* **High variance** → overfitting.
* Goal: balance both.

### Q12: Difference between Filter, Wrapper, and Embedded methods.

* Filter: Uses statistical tests (Chi-Square, Correlation).
* Wrapper: Model-based selection (RFE).
* Embedded: Regularization, tree importance.

### Q13: PCA vs Feature Selection?

* PCA: Creates new features (loss of interpretability).
* FS: Selects from existing features (keeps interpretability).

### Q14: Why does regularization help reduce overfitting?

* It penalizes large weights → simpler, more generalizable models.

---

## ✅ Interview-Ready Answer Template

**Question**: How do you select features in your projects?

**Answer**:

1. Remove constant & quasi-constant features.
2. Remove highly correlated features.
3. Apply univariate tests (Chi-Square, ANOVA, MI).
4. Use embedded models (Lasso, RF, XGBoost).
5. Validate with cross-validation.

---

## 📌 Quick Formulas Reference

* **VIF**: $VIF_i = 1/(1 - R_i^2)$
* **Mutual Information**: $I(X;Y) = \sum p(x,y) \log(\frac{p(x,y)}{p(x)p(y)})$
* **Ridge**: Loss + $\lambda \sum \theta^2 $
* **Lasso**: Loss + $\lambda \sum |\theta| $
* **F1 Score**: $2 \cdot (Precision \cdot Recall)/(Precision + Recall)$
* **Adjusted R²**: $1 - (1 - R^2)(n-1)/(n-p-1)$
* **K-Fold CV Error**: Avg of fold errors.

---

This guide covers **20+ interview-ready questions with formulas** for **feature selection, model evaluation, and overfitting control**.
