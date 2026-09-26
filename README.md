# Real Estate Valuation & Machine Learning Pipeline

An end-to-end data science and regression project that processes, cleans, and models a large-scale real estate dataset (160k+ records) to predict property valuations using Multiple Linear Regression.

---

## 🛠️ Engineering Challenges & Solutions

Real-world datasets are inherently messy. This project addresses several core data integrity issues:
1. **Fragmented Area Columns:** Real estate records split size metrics across `Super Area`, `Carpet Area`, and `Plot Area` with inconsistent formatting. Developed a dynamic fallback ETL pipeline to consolidate these columns into a single unified `Area` metric.
2. **Data Type & Text Sanitization:** Built robust helper functions using Python regex and string manipulation to strip units (e.g., `"sqft"`) and parse complex numerical monetary values.
3. **Outlier Filtering & Target Correction:** Debugged mislabeled target variables (differentiating between rate-per-sqft and absolute property amounts) and filtered out extreme data entry errors to stabilize model convergence.

---

## 📊 Model Performance

* **Algorithm:** Multiple Linear Regression (Scikit-Learn)
* **Dataset Size:** 166,000+ processed rows
* **Evaluation Metric ($R^2$ Score):** **0.47** (explains 47% of price variance based purely on area and structural cleaning)

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with the required data science libraries:
```bash
pip install pandas numpy scikit-learn matplotlib
```
