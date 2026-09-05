# 🏠 Boston Housing Price Prediction

## 📌 Project Overview
This project applies machine learning techniques to the **Boston Housing dataset** to predict the **median value of owner-occupied homes (MEDV)**.  
It demonstrates a complete **end-to-end data science workflow**, from data preprocessing and feature engineering to model training, evaluation, and deployment via a **Streamlit interactive dashboard**.

---

## 🎯 Objectives
- Build **simple and multiple linear regression models** to predict housing prices  
- Evaluate model performance using:
  - R²
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
- Perform **residual analysis** to validate regression assumptions
- Interpret regression coefficients to identify **key drivers of housing prices**

---

## ⚙️ Workflow

### 1️⃣ Data Preparation
- **Dataset**: `HousingData.csv` (506 rows, 14 columns)

**Missing Values**
- Continuous variables → median imputation
- Categorical variable (`CHAS`) → mode imputation

**Outlier Handling**
- Log transforms: `CRIM`, `ZN`, `LSTAT`
- Winsorization: `RM` capped at 95th percentile

---

### 2️⃣ Feature Engineering
Final model features:
- `LSTAT_log`
- `RM_winsor`
- `ZN_log`
- `PTRATIO`
- `CRIM_log`

---

### 3️⃣ Model Training
Pipeline:
- PolynomialFeatures (interaction_only)
- StandardScaler
- RidgeCV

**Performance**
- R² ≈ 0.785
- RMSE ≈ 3.97
- MAE ≈ 2.47
- MAPE ≈ 12.2%

Model saved as `ridge_model.pkl`

---

### 4️⃣ Deployment
Interactive Streamlit dashboard with:
- Slider-based inputs
- Real-time predictions
- Feature visualizations
- Sensitivity analysis

---

## 🚀 Running the App

```bash
pip install streamlit scikit-learn pandas numpy seaborn matplotlib joblib
streamlit run app.py
```

---

## 📂 Project Structure
```
├── HousingData.csv
├── ridge_model.pkl
├── app.py
├── README.md
```

---

## ✅ Conclusion
This project turns a complete machine learning workflow into a deployable and interpretable application suitable for scenario analysis and stakeholder use.
