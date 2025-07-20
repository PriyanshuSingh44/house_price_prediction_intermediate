# 🏠 House Price Prediction (Intermediate Level)

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Ensemble%20Model-0C5A9A?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC2D01?logo=xgboost&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

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
├── .gitignore
├── LICENSE
├── README.md   # Project documentation (this file)
├── requirements.txt             
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

🔹 To run the **XGBoost model notebook**:

```bash
jupyter notebook xgboost_model.ipynb
# OR to run all cells at once:
jupyter nbconvert --to notebook --execute --inplace xgboost_model.ipynb
```

🔸 To run the Random Forest model notebook:

```bash
jupyter notebook random_forest_model.ipynb
# OR to run all cellls at once:
jupyter nbconvert --to notebook --execute --inplace random_forest_model.ipynb
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
