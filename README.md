# Delivery ETA Prediction & Dark Store Analysis

An end-to-end machine learning project for predicting delivery time from operational, temporal, environmental, and geographic features, followed by a separate dark-store proximity analysis across Blinkit, Zepto, and Instamart.

> The ETA model is trained for restaurant-to-customer delivery.
## Overview

## Live Demo

[![Live App](https://img.shields.io/badge/Live%20Demo-Streamlit-34C363?style=for-the-badge&logo=streamlit&logoColor=white)](https://food-delivery-eta-prediction.streamlit.app/)

Try the deployed **Delivery ETA Predictor** directly in your browser.

[Open Live App](https://food-delivery-eta-prediction.streamlit.app/)

The project follows a complete ML workflow:

```text
Data → Cleaning → EDA → Feature Engineering
     → Leakage-safe Preprocessing → Model Comparison
     → Cross-Validation & Tuning → Evaluation
     → Feature Importance & SHAP → Final XGBoost
     → Streamlit Deployment
```

<!-- Add the main Excalidraw workflow diagram here -->

## Dataset

The project uses a Kaggle food-delivery dataset containing **45,584 rows and 20 original columns**.

**Main Delivery Dataset:** [Kaggle](https://www.kaggle.com/datasets/afreeenarshi/zomato)

The target is:

```text
Time_taken (min)
```

Key inputs include delivery-person information, restaurant/customer coordinates, weather, traffic, vehicle condition, order type, multiple deliveries, festival, city, order time, pickup time, and order date.

## Data Preparation

### Inspection

The raw data was inspected for:

- shape, columns and data types
- missing values
- duplicates
- numerical and categorical distributions
- inconsistent time formats
- suspicious geographic and demographic values

### Cleaning

- Removed `ID` and `Delivery_person_ID`
- Converted mixed `Time_Orderd` and `Time_Order_picked` values into `Order_Hour` and `Picked_Hour`
- Converted `Order_Date` into `Day_of_Week`
- Median-imputed missing numerical values
- Corrected negative restaurant latitudes using absolute values
- Recalculated geographic distance after coordinate correction
- Removed remaining incomplete rows
- Reset the index

After cleaning:

| Metric | Value |
|---|---:|
| Rows | 42,508 |
| Columns | 18 |
| Missing values | 0 |
| Duplicate rows | 0 |

> [!WARNING]
> The source data contains anomalies such as age/rating combinations that appear systematically encoded. These were retained rather than manually changed because their original meaning could not be verified.

## Exploratory Data Analysis

EDA was used to understand relationships between delivery time and the available features.

Important observations:

- Multiple deliveries were associated with higher delivery times.
- Higher delivery-person ratings were associated with lower delivery times.
- Vehicle condition showed a negative relationship with delivery time.
- Order and pickup hours captured time-of-day effects.
- Traffic, weather, festival status, city type, and distance showed meaningful differences in ETA.
- Traffic Jam and festival conditions were among the clearest high-delay signals.

## Feature Engineering

### Haversine Distance

Restaurant and customer locations are represented by latitude and longitude, so geographic distance was calculated using the **Haversine formula**.

```text
Restaurant coordinates
        ↓
   Haversine formula
        ↓
     distance_km
```

The resulting distance feature ranged from approximately **1.47 km to 20.97 km**, with a median of **9.22 km**.

Distance had a correlation of approximately **0.323** with delivery time, showing that it matters but does not explain ETA on its own.

### Time Features

The project also derives:

- `Order_Hour`
- `Picked_Hour`
- `Day_of_Week`

These capture time-of-day and weekly operational patterns.

## Preprocessing

An **80/20 train-test split** was used with `random_state=42`.

The preprocessing workflow uses `ColumnTransformer`:

| Feature type | Transformation |
|---|---|
| Numerical | `StandardScaler` |
| Categorical | `OneHotEncoder(handle_unknown="ignore")` |

The preprocessing and model are wrapped in a single Scikit-learn `Pipeline`.

> [!IMPORTANT]
> The pipeline is fitted only on the training data. This keeps the test set unseen during preprocessing and helps prevent data leakage. The same fitted transformations are then automatically applied during inference.

## Model Development

Several regression approaches were considered, including linear models, tree-based models, ensemble methods, and XGBoost.

The main model comparison showed that nonlinear ensemble models substantially outperformed the linear baseline.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 4.70 | 5.92 | 0.602 |
| Decision Tree | 4.05 | 5.26 | 0.685 |
| Gradient Boosting | 3.50 | 4.36 | 0.785 |
| Random Forest | 3.09 | 3.85 | 0.832 |
| XGBoost | 3.14 | 3.91 | 0.827 |
| Tuned Random Forest | 3.04 | 3.77 | 0.838 |
| **Tuned XGBoost** | **3.01** | **3.76** | **0.840** |

### Cross-Validation & Hyperparameter Tuning

Randomized hyperparameter search with **5-fold cross-validation** was used for Random Forest and XGBoost.

The final XGBoost configuration included:

```text
n_estimators     = 500
max_depth        = 7
learning_rate    = 0.02
subsample        = 0.8
colsample_bytree = 1.0
min_child_weight = 7
reg_lambda       = 1
```

## Final Model

**Tuned XGBoost** was selected as the final model.

Test performance:

| Metric | Result |
|---|---:|
| MAE | **3.01 min** |
| RMSE | **3.76 min** |
| R² | **0.840** |

The final model is stored as:

```text
final_eta_model.pkl
```

The `.pkl` contains the complete preprocessing + model pipeline, so a separate scaler or encoder file is not required.

## Feature Importance & SHAP

Feature importance was used to identify influential transformed features, while **SHAP** was used for global and local model explainability.

Important signals included:

- road traffic density
- multiple deliveries
- weather conditions
- delivery-person rating
- festival conditions
- vehicle condition
- distance

SHAP helps explain not only which features matter, but also **how individual feature values push a prediction higher or lower**.

## Dark Store Analysis

A separate geographic extension was performed using dark-store locations from:

### Dark Store Datasets

| Platform | Dataset |
|---|---|
| Blinkit | [Kaggle](https://www.kaggle.com/datasets/afreeenarshi/blinkit-darkstore-dataset) |
| Zepto | [Kaggle](https://www.kaggle.com/datasets/afreeenarshi/zepto-darkstore-dataset) |
| Instamart | [Kaggle](https://www.kaggle.com/datasets/afreeenarshi/instamart-darkstore-dataset) |


- Blinkit
- Zepto
- Instamart

Store counts:

| Platform | Stores |
|---|---:|
| Blinkit | 1,954 |
| Zepto | 1,089 |
| Instamart | 1,038 |

Customer-to-store candidates were searched within a **4 km radius** using geographic nearest-neighbor search with Haversine distance.

Results:

| Platform | Candidate pairs |
|---|---:|
| Blinkit | 115,218 |
| Instamart | 77,156 |
| Zepto | 94,691 |
| **Total** | **287,065** |

The nearest store per customer and platform was then selected for distance and coverage comparison.

> [!WARNING]
> The restaurant-delivery ETA model was tested with dark-store coordinates, but the resulting quick-commerce ETA predictions were not operationally realistic. Therefore, the project does **not** claim that this model is a valid quick-commerce ETA model. A dedicated quick-commerce ETA model would require appropriate quick-commerce delivery data and target labels.

<!-- Add the dark-store Excalidraw diagram here -->

## Streamlit Application

The trained pipeline was integrated into a Streamlit dashboard.

The application:

1. collects delivery details from the user
2. creates a single-row input DataFrame
3. passes it to `final_eta_model`
4. returns the estimated delivery time in minutes

```python
model = joblib.load("final_eta_model.pkl")

prediction = model.predict(input_data)[0]
```

This turns the notebook-trained model into an interactive prediction interface.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Geographic Search | BallTree, Haversine |
| Model Persistence | Joblib |
| Dashboard | Streamlit |
| Development | Kaggle, VS Code, WSL |

> [!NOTE]
> The saved model was trained with **scikit-learn 1.6.1**. Keeping the deployment environment compatible with the training environment is important when loading serialized Scikit-learn pipelines.

## Project Structure

```text
Delivery-ETA-Prediction-Dark-Store-Analysis/
│
├── app.py
├── final_eta_model.pkl
├── delivery-eta-prediction-dark-store-analysis.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── config.toml
```

## Run Locally

### Clone the repository

```bash
git clone https://github.com/arshiafreen090/Delivery-ETA-Prediction-Dark-Store-Analysis.git
cd Delivery-ETA-Prediction-Dark-Store-Analysis
```

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

## Resources

- **Live Application:** [Delivery ETA Predictor](https://food-delivery-eta-prediction.streamlit.app/)
- **Main Dataset:** [Food Delivery Dataset](https://www.kaggle.com/datasets/afreeenarshi/zomato)
- **Blinkit Dark Store Dataset:** [Kaggle](https://www.kaggle.com/datasets/afreeenarshi/blinkit-darkstore-dataset)
- **Zepto Dark Store Dataset:** [Kaggle](https://www.kaggle.com/datasets/afreeenarshi/zepto-darkstore-dataset)
- **Instamart Dark Store Dataset:** [Kaggle](https://www.kaggle.com/datasets/afreeenarshi/instamart-darkstore-dataset)
