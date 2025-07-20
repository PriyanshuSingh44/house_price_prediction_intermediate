# 🏠 House Price Prediction (Intermediate Level)

This project predicts house prices using two separate machine learning models: **XGBoost Regressor** and **Random Forest Regressor**. The implementation is based on what was learned in the Kaggle Intermediate Machine Learning course. Each model is trained in its own script, and the code is kept simple and easy to follow (not modularized yet).

---

## 📂 Project Structure

```
house_price_prediction_intermediate/
│
├── data/
│   ├── train.csv
|
├── Notebooks/
|   ├── XGBoost.ipynb  # Trains and evaluates using XGBRegressor
|   ├── RandomForest.ipynb # # Trains and evaluates using RandomForestRegressor
|
├── README.md              # Project documentation (this file)
```

---

## 📌 Features

- Two separate scripts for training:
  - `xgboost_model.ipynb`
  - `random_forest_model.ipynb`
- Each script:
  - Loads the data
  - Preprocesses features
  - Trains a regression model
  - Evaluates using **Mean Absolute Error (MAE)**

---

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

Typical libraries used:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`

---

## 🚀 Usage

### 🔹 To run the XGBoost model:

```bash
python xgboost_model.ipynb
```

### 🔸 To run the Random Forest model:

```bash
python random_forest_model.ipynb
```

---

## 🧠 Models

### XGBoost Regressor

```python
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X_train, y_train)
```

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
```

---

## 👨‍💻 Author

**Priyanshu Singh**  
📧 priyanshusingh442000@gmail.com  
🔗 [GitHub: PriyanshuSingh44](https://github.com/PriyanshuSingh44)

---

## 📘 Reference

Based on [Kaggle's Intermediate Machine Learning Course](https://www.kaggle.com/learn/intermediate-machine-learning)
