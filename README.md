# 🌧️ Rain Prediction using Random Forest

This machine learning project predicts whether it will rain or not using historical weather data. The goal is to classify weather conditions into "rain" or "no rain" using temperature, humidity, pressure, and other meteorological variables.

This project walks through the full ML pipeline: data cleaning, feature engineering, model training, evaluation, and interpretation — resulting in a high-performing classifier with excellent generalization.

---

## 📁 Dataset

- **Source**: [Kaggle - Synthetic Weather Forecasting Dataset](https://www.kaggle.com/)
- **File**: `weather_forecast_data.csv`
- **Rows**: 2,500 samples
- **Target**: `Rain` (`'rain'` or `'no rain'`)
- **Features**:
  - `Temperature` (°C)
  - `Humidity` (%)
  - `Wind_Speed` (m/s)
  - `Cloud_Cover` (%)
  - `Pressure` (hPa)

---

## 🧠 Feature Engineering

Initially, the Random Forest model showed signs of **overfitting** due to limited raw features and strong correlations.  
To improve generalization, I generated new features to capture meaningful patterns in the data:

### 🔨 Engineered Features:
- `Temp_Humidity_Ratio` – distinguishes dry heat from humid weather
- `Pressure_Drop_Risk` – flags low-pressure conditions more likely to cause rain
- `Feels_Like_Index` – combines temperature and humidity
- `Is_Cloudy` – binary flag for cloud cover > 50%
- `High_Humidity` – binary flag for humidity > 85%
- **Rolling statistics (window = 3):**
  - Averages: `Temperature_MA3`, `Humidity_MA3`, `Pressure_MA3`, `Cloud_Cover_MA3`
  - Volatility: `Humidity_STD3`, `Pressure_STD3`

📈 These features helped reduce overfitting and improved model precision and recall on both train and test data.

---

## 🔍 Model Overview

- **Algorithm**: Random Forest Classifier (scikit-learn)
- **Hyperparameters**:
  - `n_estimators=100`
  - `max_depth=3`
  - `min_samples_split=10`
  - `min_samples_leaf=4`
- **Final Model File**: `rain_predictor_rf.pkl`

---

## 🧪 Evaluation

This model was trained and validated using a train/test split, a separate test dataset, and 5-fold cross-validation to ensure generalization.

---

### 🏋🏽‍♂️ Training Results

On the training split:

| Metric         | Value |
|----------------|--------|
| Accuracy       | **97%**
| Precision (no rain) | 0.97 |
| Recall (no rain)    | 1.00 |
| F1-Score (no rain)  | 0.98 |
| Precision (rain)    | 1.00 |
| Recall (rain)       | 0.83 |
| F1-Score (rain)     | 0.91 |

> The model achieved high precision on both classes and caught 83% of all rainy cases during training.

---

### 🧾 Test Set Results (on `weather_forecast_test_data.csv`)

On 498 new, unseen samples:

| Metric         | Value |
|----------------|--------|
| Accuracy       | **99%**
| Precision (no rain) | 0.99 |
| Recall (no rain)    | 1.00 |
| F1-Score (no rain)  | 0.99 |
| Precision (rain)    | 0.98 |
| Recall (rain)       | 0.90 |
| F1-Score (rain)     | 0.94 |

📌 Only 6 false negatives on rain detection — overall strong generalization.

---

### 🔁 5-Fold Cross-Validation

| Fold | Accuracy |
|------|----------|
| 1    | 0.975    |
| 2    | 0.982    |
| 3    | 0.979    |
| 4    | 0.978    |
| 5    | 0.976    |

📊 **Average Cross-Validation Accuracy: 97.8%**

> Cross-validation confirms the model is consistent and robust across different data splits.

---

### 📊 Feature Importance

Top features by importance:
1. `Temp_Humidity_Ratio`
2. `Pressure`
3. `Humidity`
4. `Is_Cloudy`
5. `Feels_Like_Index`

> Adding these engineered features significantly improved the model's ability to detect rain, raising recall from ~75% to 90% and reducing false negatives.

---

## 📂 Files Included

- `weather_forecast_data.csv` — full dataset
- `weather_forecast_test_data.csv` — test split
- `notebook.ipynb` — end-to-end code notebook
- `README.md` — project documentation

---


## 👨🏽‍💻 Author

Made with 💡 and ☕ by **Tshepang Ramaoka**
