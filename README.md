# Car Price Prediction

This project predicts the selling price of a used car using basic machine learning regression models.

The main work lives in [Car_Price_Prediction.ipynb](C:/Users/Hp_Ed/OneDrive/Desktop/Car-Price-Prediction/Car_Price_Prediction.ipynb), where the full workflow is written step by step in notebook form.

## Project Goal

The notebook tries to estimate a car's `Selling_Price` from features such as:

- `Year`
- `Present_Price`
- `Kms_Driven`
- `Fuel_Type`
- `Seller_Type`
- `Transmission`
- `Owner`

Two regression models are used:

- `LinearRegression`
- `Lasso`

## Dataset Overview

From the notebook output:

- Total rows: `301`
- Total columns: `9`
- Missing values: `0`

Original dataset columns:

- `Car_Name`
- `Year`
- `Selling_Price`
- `Present_Price`
- `Kms_Driven`
- `Fuel_Type`
- `Seller_Type`
- `Transmission`
- `Owner`

The target column is:

- `Selling_Price`

## Workflow

The notebook follows a simple machine learning pipeline.

### 1. Import Libraries

The project imports:

- `pandas` for data handling
- `matplotlib.pyplot` for plotting
- `train_test_split` for splitting the dataset
- `LinearRegression` and `Lasso` from scikit-learn
- `metrics` for model evaluation

### 2. Load the Dataset

The notebook loads the CSV file with:

```python
car_dataset = pd.read_csv('/content/car data.csv')
```

This path shows the notebook was likely run in Google Colab. If you run it locally, you will usually need to change it to something like:

```python
car_dataset = pd.read_csv('car data.csv')
```

### 3. Explore the Data

Before modeling, the notebook checks:

- the first few rows with `head()`
- dataset shape with `shape`
- column types with `info()`
- missing values with `isnull().sum()`
- summary statistics with `describe()`

This step is important because it confirms:

- which columns are numeric
- which columns are categorical
- whether data cleaning is needed
- the rough range of each feature

### 4. Encode Categorical Features

Machine learning models in scikit-learn need numeric input, so categorical columns are converted to numbers.

The notebook applies this mapping:

- `Fuel_Type`: `Petrol -> 0`, `Diesel -> 1`, `CNG -> 2`
- `Seller_Type`: `Dealer -> 0`, `Individual -> 1`
- `Transmission`: `Manual -> 0`, `Automatic -> 1`

This allows the models to train on the dataset without one-hot encoding.

### 5. Split Features and Target

The notebook creates:

- `x` as input features
- `y` as the target

Code used:

```python
x = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
y = car_dataset['Selling_Price']
```

`Car_Name` is removed because it is text and is not used as a feature in this notebook.

### 6. Train-Test Split

The data is split into training and testing sets:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=2
)
```

What this means:

- `90%` of the data is used for training
- `10%` is used for testing
- `random_state=2` keeps the split reproducible

### 7. Train the Linear Regression Model

The notebook creates and trains a linear regression model:

```python
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
```

After training, predictions are generated for:

- training data
- test data

### 8. Evaluate Linear Regression

The notebook uses `R² score` to measure performance.

Saved notebook results:

- Training R²: `0.8799451660493711`
- Test R²: `0.8365766715027051`

Interpretation:

- a higher R² means the model explains more of the variation in selling price
- the train and test scores are fairly close, which suggests the model is learning useful patterns without obvious severe overfitting

The notebook also plots:

- actual prices vs predicted prices for training data
- actual prices vs predicted prices for test data

If predictions are strong, the scatter points should roughly align along a diagonal pattern.

### 9. Train the Lasso Model

The second model is:

```python
lass_reg_model = Lasso()
lass_reg_model.fit(x_train, y_train)
```

Lasso is a regularized linear model. It can sometimes perform better than plain linear regression when some features are less useful or when we want a simpler model.

### 10. Evaluate Lasso

Saved notebook results:

- Reported training R²: `0.8799451660493711`
- Test R²: `0.8709167941173195`

The Lasso test score is slightly better than the linear regression test score in the current notebook output.

## Important Notes About the Current Notebook

These are worth knowing if you maintain or improve this project.

### 1. Repeated Encoding Cell

The categorical replacement step appears twice in the notebook. It does not break the workflow here, but it is redundant and can be cleaned up.

### 2. Lasso Training Score Cell Uses the Wrong Predictions

In the Lasso section, the notebook prints the training R² using `training_data_prediction`, which was created earlier from the linear regression model.

That means the displayed Lasso training R² is not actually a fresh Lasso training evaluation.

The current cell should ideally be changed to something like:

```python
lasso_training_prediction = lass_reg_model.predict(x_train)
error_score = metrics.r2_score(y_train, lasso_training_prediction)
print("R squared Error : ", error_score)
```

So the Lasso training result matches the model being evaluated.

### 3. Dataset File Is Not Included in This Repo Snapshot

The notebook expects a file named `car data.csv`, but the repository currently contains only the notebook.

To run the project successfully, make sure that CSV file is added to the repo or placed in the correct path.

## End-to-End Flow in Plain Words

If we explain the whole notebook like a developer would build it, the workflow is:

1. Read the car dataset.
2. Inspect the data structure and confirm there are no missing values.
3. Convert text categories into numeric values.
4. Separate inputs from the target price.
5. Split the data into training and testing sets.
6. Train a baseline `LinearRegression` model.
7. Measure how well it predicts prices.
8. Train a `Lasso` model as a second comparison.
9. Compare both models using R² and scatter plots.

## Tech Stack

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab style workflow

## Suggested Repository Structure

If you want to make this repo easier for others to use, a clean next version could look like:

```text
Car-Price-Prediction/
|-- Car_Price_Prediction.ipynb
|-- car data.csv
|-- README.md
|-- requirements.txt
```

## Suggested Improvements

Some practical upgrades you can make later:

- add the dataset file to the repo
- fix the Lasso training evaluation cell
- remove the duplicate encoding cell
- create a `requirements.txt`
- add feature engineering such as car age: `current_year - Year`
- compare more models like `RandomForestRegressor`
- save the trained model with `joblib` or `pickle`
- build a small prediction app with Streamlit or Flask

## Conclusion

This repository is a beginner-friendly machine learning project that demonstrates the full regression workflow:

- loading data
- preprocessing categorical features
- training models
- evaluating predictions

It is a solid starting point for learning how price prediction projects are structured, and with a few cleanup steps it can become a stronger portfolio project as well.
