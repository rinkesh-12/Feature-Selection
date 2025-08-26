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
