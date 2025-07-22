# 🏠 House Price Prediction (Intermediate Level)

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?logo=matplotlib&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Ensemble%20Model-0C5A9A?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC2D01?logo=xgboost&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

This project predicts house prices using two separate machine learning models: **XGBoost Regressor**, **XGBoost Regressor with GridSearchCV** and **Random Forest Regressor**. The implementation is based on what was learned in the Kaggle Intermediate Machine Learning course. Each model is trained in its own script, and the code is kept simple and easy to follow (not modularized yet).

---

## 📂 Project Structure

```
house_price_prediction_intermediate/
│
├── data/
|   ├── test.csv
│   ├── train.csv
|
├── Notebooks/
|   ├── RandomForest.ipynb  # Trains and evaluates using RandomForestRegressor
|   ├── XGBoost.ipynb  # Trains and evaluates using XGBRegressor
|   ├── xgboost_gridsearchCV.ipynb  # Used xgboost regressor with GridSearchCV for parameter tuning
|
├── .gitignore
├── LICENSE
├── README.md   # Project documentation (this file)
├── requirements.txt             
```

---

## 📌 Features

- Three separate scripts for training:
  - `RandomForest.ipynb`
  - `XGBoost.ipynb`
  - `xgboost_gridsearchCV.ipynb`
- Each script:
  - Loads the data
  - Preprocesses features
  - Finetune for best parameters (only xgboost_gridsearchCV.ipynb)
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
- `matplotlib`

---

## 🚀 Usage

🔹 To run the **XGBoost model notebook**:

```bash
jupyter notebook XGBoost.ipynb
# OR to run all cells at once:
jupyter nbconvert --to notebook --execute --inplace XGBoost.ipynb
```

🔹 To run the **XGBoost with GridSearchCV model notebook**:

```bash
jupyter notebook xgboost_gridsearchCV.ipynb
# OR to run all cells at once:
jupyter nbconvert --to notebook --execute --xgboost_gridsearchCV.ipynb
```

🔸 To run the Random Forest model notebook:

```bash
jupyter notebook RandomForest.ipynb
# OR to run all cellls at once:
jupyter nbconvert --to notebook --execute --inplace RandomForest.ipynb
```

---

## 🧠 Models

### XGBoost Regressor

```python
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X_train, y_train)
```

### XGBoost Regressor with GridSearchCV

```python
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

model = XGBRegressor()
param_grid = {                                  # these are sample values (you can say that)
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.7, 1.0]
}
grid = GridSearchCV()
grid.fit(X_train, y_train)   # We don't need to explicitly fit the model — GridSearchCV does that for us.
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
