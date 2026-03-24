# BMW Price Prediction using Random Forest Regressor

This project is a machine learning regression model built in Python to predict used BMW car prices based on vehicle features such as year, transmission, mileage, fuel type, tax, miles per gallon, and engine size. It uses pandas for data processing and scikit-learn's Random Forest Regressor to generate predicted prices, evaluate model performance, and export a comparison CSV.


## Features

- Loads BMW car data from a CSV file
- Encodes categorical columns using vectorised binary conversion
- Splits data into training and test sets
- Trains a Random Forest Regressor with 1000 estimators
- Evaluates model using MAE, RMSE, and R²
- Exports a comparison CSV of original vs predicted prices


## Installation

1. Clone or download the project
2. Install dependencies:

```bash
pip install -r requirements.txt
```


## Requirements

```txt
pandas
scikit-learn
```


## Dataset

The model expects a CSV file named `dataset.csv` in the same directory. The dataset should contain the following columns:

- `model` — car model name
- `year` — year of manufacture
- `price` — listing price (target variable)
- `transmission` — `Automatic` or `Manual`
- `mileage` — mileage of the car
- `fuelType` — `Diesel`, `Petrol`, or other
- `tax` — road tax amount
- `mpg` — miles per gallon
- `engineSize` — engine size in litres


## How It Works

### 1. Load the dataset
The dataset is read from `dataset.csv` into a pandas DataFrame.

### 2. Encode categorical columns
The `binarise()` function converts:
- `transmission` → `1` for Automatic, `0` for all others
- `fuelType` → `1` for Diesel, `0` for all others

This uses vectorised pandas operations for efficiency.

### 3. Prepare features and target
The age column is a new feature added to factor in how old the car is.
The `model` column is stored separately and dropped from training features. The `price` column is used as the target variable.

### 4. Train the model
An 70/30 train-test split is applied. A `RandomForestRegressor` is trained on the training set using 1000 estimators and all available CPU cores.

### 5. Evaluate performance
The model is evaluated on the test set using:
- **MAE** — Mean Absolute Error

### 6. Export results
A CSV file named `prediction-original-prices.csv` is generated containing:
- `model` — car model name as index
- `originalPrice` — actual price from the dataset
- `predictedPrice` — model's predicted price


## Running the Project

```bash
python bmw_price_pred.py
```

## Output CSV Example

| model | originalPrice | predictedPrice |

| 5 Series | 19995 | 20110.42 |
| 3 Series | 14500 | 14287.31 |
| X5 | 23999 | 24105.76 |


## Possible Improvements

- Tune hyperparameters using GridSearchCV or RandomizedSearchCV
- Add data visualisations for predicted vs actual prices