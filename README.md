# Real Estate Valuation & Machine Learning Pipeline

An end-to-end data science project that processes, cleans, and models real estate data (10k+ records) to predict property valuations and classify market tiers using **Linear Regression** and **Logistic Regression**.

---

## 🛠️ Engineering Challenges & Solutions

Real-world datasets are inherently messy. This project tackles core data integrity issues:
1. **Multi-Source Area Consolidation:** Handled fragmented area columns (`SUPER_SQFT`, `BUILTUP_SQFT`, `AREA`) by developing a dynamic fallback pipeline to unify size metrics.
2. **Text & Unit Sanitization:** Built robust helper functions using Python string manipulation and regex to strip units (e.g., "sqft") and parse complex string-based numerical entries.
3. **Outlier Filtering & Range Sanitization:** Cleaned anomalous data entry errors and applied rigorous value filtering to stabilize model convergence and prevent data skew.

---

## 📊 Models & Methodology

This project implements two core machine learning paradigms:

1. **Regression (`model.py`)**: 
   * **Algorithm:** Multiple Linear Regression (`scikit-learn`)
   * **Target:** Continuous property pricing.
   * **Evaluation Metric:** $R^2$ Score & Mean Squared Error (MSE).
   * **Visualization:** Generates scatter plots paired with smooth linear regression lines.

2. **Classification (`classifier.py`)**:
   * **Algorithm:** Logistic Regression (`scikit-learn`)
   * **Target:** Binned market tier prediction (Budget vs. Luxury classification based on median price thresholds).
   * **Evaluation Metric:** Accuracy and Classification Report (Precision/Recall).
   * **Visualization:** Plots actual test points alongside the signature Sigmoid probability curve.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with the required data science libraries:
```bash
pip install pandas numpy scikit-learn matplotlib kagglehub joblib