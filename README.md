# Heart Disease Prediction App

An end-to-end machine learning project for predicting the presence of heart disease from patient clinical attributes. The project explores and preprocesses the UCI Heart Disease dataset, compares multiple classification algorithms, tunes their hyperparameters, evaluates them with several classification metrics, interprets the selected model with SHAP, and serves the final Random Forest pipeline through a Streamlit web application.

> **Disclaimer:** This project is intended for educational and portfolio purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Project Overview

The original target column, `num`, contains values from 0 to 4. For this project it is converted into a binary classification target:

- `0` -> No heart disease
- `1` -> Heart disease present (`num > 0`)

The dataset used in the notebook contains **920 records**. After feature selection, **11 input features** are used to train the models. The data is split into **80% training data (736 rows)** and **20% test data (184 rows)** using a stratified split with `random_state=42`.

The workflow covers:

1. Data loading and inspection
2. Missing-value analysis
3. Data cleaning
4. Exploratory data analysis (EDA)
5. Feature selection
6. Train/test splitting
7. Preprocessing with Scikit-learn pipelines
8. Hyperparameter tuning with `GridSearchCV`
9. Comparison of four classification algorithms
10. Evaluation using multiple metrics
11. Model interpretation with SHAP
12. Error analysis of the selected model
13. Model serialization with Joblib
14. Deployment through Streamlit

## Dataset

The project uses the **UCI Heart Disease dataset**, stored locally as:

```text
heart_disease_uci.csv
```

The raw dataset contains 920 observations and 16 columns. The original `num` target distribution in the notebook is:

| `num` | Count |
|---:|---:|
| 0 | 411 |
| 1 | 265 |
| 2 | 109 |
| 3 | 107 |
| 4 | 28 |

After converting the problem to binary classification:

| Target | Meaning | Count |
|---:|---|---:|
| 0 | No heart disease | 411 |
| 1 | Heart disease present | 509 |

## Features Used

The final model uses the following 11 features:

| Feature | Description | Type |
|---|---|---|
| `age` | Age | Numerical |
| `trestbps` | Resting blood pressure | Numerical |
| `chol` | Serum cholesterol | Numerical |
| `thalch` | Maximum heart rate achieved | Numerical |
| `oldpeak` | ST depression induced by exercise relative to rest | Numerical |
| `sex` | Sex | Categorical |
| `cp` | Chest pain type | Categorical |
| `fbs` | Fasting blood sugar indicator | Categorical |
| `restecg` | Resting electrocardiographic result | Categorical |
| `exang` | Exercise-induced angina | Categorical |
| `slope` | Slope of the peak exercise ST segment | Categorical |

The columns `id`, `num`, `ca`, `thal`, and `dataset` are excluded from the model features. The newly created binary `target` column is used as the prediction target.

## Data Cleaning and Exploratory Analysis

The notebook first examines the dataset shape, data types, class distribution, and missing values. Missingness is also investigated by dataset source.

Zero values in `trestbps` and `chol` are treated as missing values because they are not meaningful measurements for these variables:

```python
df['trestbps'] = df['trestbps'].replace(0, np.nan)
df['chol'] = df['chol'].replace(0, np.nan)
```

The exploratory analysis includes:

- Original and binary target distributions
- Missing-value rates by data source
- Histograms for resting blood pressure and cholesterol
- Box plots of numerical features against the target
- Categorical feature distributions against the target
- Correlation heatmap for numerical features

## Preprocessing Pipeline

Preprocessing is handled inside a Scikit-learn `ColumnTransformer`, which keeps data preparation and model inference in a single reproducible pipeline.

### Numerical features

```text
age, trestbps, chol, thalch, oldpeak
```

Processing steps:

- Missing values -> median imputation
- Feature scaling -> `StandardScaler`

### Categorical features

```text
sex, cp, fbs, restecg, exang, slope
```

Processing steps:

- Missing values -> most-frequent imputation
- Encoding -> `OneHotEncoder(handle_unknown='ignore')`

This preprocessing pipeline is attached directly to every classifier, helping prevent inconsistent preprocessing between training and inference.

## Models Compared

Four classification algorithms are trained and tuned:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost Classifier**

Hyperparameter tuning is performed with **5-fold cross-validation** using `GridSearchCV`, with **F1 score** as the optimization metric.

## Hyperparameter Tuning Results

### Logistic Regression

Best parameters:

```text
C = 0.1
l1_ratio = 0.0
solver = saga
```

Best cross-validation F1 score: **0.8254**

### Decision Tree

Best parameters:

```text
max_depth = 5
min_samples_leaf = 4
min_samples_split = 10
```

Best cross-validation F1 score: **0.7841**

### Random Forest

Best parameters:

```text
max_depth = 5
max_features = sqrt
min_samples_split = 5
n_estimators = 200
```

Best cross-validation F1 score: **0.8367**

### XGBoost

Best parameters:

```text
learning_rate = 0.01
max_depth = 3
n_estimators = 300
subsample = 1.0
```

Best cross-validation F1 score: **0.8316**

## Model Evaluation

The tuned models are evaluated on the held-out test set using Accuracy, Precision, Recall, F1 score, and ROC-AUC.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.793 | 0.796 | 0.843 | 0.819 | 0.888 |
| Decision Tree | 0.777 | 0.780 | 0.833 | 0.806 | 0.843 |
| **Random Forest** | **0.793** | **0.796** | **0.843** | **0.819** | **0.907** |
| XGBoost | 0.788 | 0.779 | **0.863** | 0.819 | 0.903 |

The notebook also compares the models using confusion matrices and ROC curves.

## Final Model

**Random Forest** is selected as the final model. It achieved the highest cross-validation F1 score (**0.8367**) among the models tested and the highest test ROC-AUC (**0.907**). Its test-set accuracy was **79.3%** and its F1 score was **0.819**.

The selected object is the complete fitted Scikit-learn pipeline, not just the Random Forest estimator. Therefore, preprocessing is automatically applied when new raw observations are passed to the model.

The fitted pipeline is saved with:

```python
final_model = rf_grid.best_estimator_
joblib.dump(final_model, "final_model.joblib")
```

## Model Explainability with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to inspect the behavior of the selected Random Forest model.

The notebook:

- Retrieves feature names after one-hot encoding
- Transforms the test set with the fitted preprocessing pipeline
- Creates a `TreeExplainer` for the Random Forest
- Generates a SHAP summary plot
- Examines an individual misclassified observation using a SHAP force plot

This adds interpretability beyond reporting predictive metrics alone.

## Error Analysis

The Random Forest model misclassified **38 of 184** test observations:

- **16 false negatives**
- **22 false positives**

The notebook also examines predicted probabilities for misclassified observations and uses SHAP to inspect an example error.

## Streamlit Application

The trained pipeline is loaded directly into the Streamlit application:

```python
classifier = joblib.load("final_model.joblib")
```

User input is converted into a one-row Pandas DataFrame with the same feature names used during training, then passed directly to the saved pipeline:

```python
prediction = classifier.predict(input_data)
```

Because preprocessing is bundled with the trained model, the Streamlit application does not need to separately perform imputation, scaling, or one-hot encoding.

## Project Structure

```text
heart-disease-prediction-app/
|
|-- app.py                    # Streamlit web application
|-- final_model.joblib        # Serialized fitted Random Forest pipeline
|-- heart_disease_uci.csv     # Dataset
|-- main.ipynb                # EDA, preprocessing, training and evaluation
|-- README.md                 # Project documentation
`-- .ipynb_checkpoints/       # Jupyter-generated checkpoints (do not commit)
```

For a cleaner repository, `.ipynb_checkpoints/` should be excluded using `.gitignore`.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Streamlit
- Jupyter Notebook

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/heart-disease-prediction-app.git
cd heart-disease-prediction-app
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

If you add a `requirements.txt` file to the repository:

```bash
pip install -r requirements.txt
```

The project currently requires packages including:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
joblib
streamlit
jupyter
```

## Running the Application

Make sure `final_model.joblib` is in the same directory as `app.py`, then run:

```bash
streamlit run app.py
```

Streamlit will start a local development server and display the application in your browser.

## Reproducing the Model

To reproduce the training workflow:

1. Place `heart_disease_uci.csv` in the project directory.
2. Open `main.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the notebook cells in order.
4. The notebook performs preprocessing, model tuning, evaluation, SHAP analysis, and error analysis.
5. The selected fitted Random Forest pipeline is exported as `final_model.joblib`.
6. Run `streamlit run app.py` to use the exported model through the web interface.

## Suggested Repository Improvements

Before publishing the repository, consider adding:

- `requirements.txt` with pinned or compatible dependency versions
- `.gitignore` to exclude `.ipynb_checkpoints/`, virtual environments, and Python cache files
- A screenshot or GIF of the Streamlit application
- A live Streamlit deployment link, if the app is deployed
- A license if you want to specify how others may use the code

A minimal `.gitignore` could contain:

```gitignore
.ipynb_checkpoints/
__pycache__/
*.py[cod]
.venv/
venv/
.env
```

## Limitations

- The model is trained and evaluated on the dataset used in this project and has not been validated as a clinical system.
- Performance on the held-out test split does not guarantee equivalent performance on new populations or real-world clinical data.
- Missing values are handled statistically through median or most-frequent imputation, which may not represent the best strategy in every deployment setting.
- The binary target combines all original disease-severity values from 1 through 4 into a single positive class.
- The current Streamlit interface accepts raw text input, so users must provide values in a form compatible with the features used during training.

## Future Improvements

Potential extensions include:

- Replace free-text categorical fields in Streamlit with dropdown/select inputs
- Add input validation and clearer feature descriptions
- Display prediction probabilities in addition to the predicted class
- Improve the UI and add explanatory information for each input
- Add model-performance visualizations to the application
- Add SHAP-based explanations for individual predictions
- Package dependencies in `requirements.txt`
- Deploy the application with Streamlit Community Cloud or another hosting platform
- Add automated tests for preprocessing and prediction behavior

## Author

**Vinay Kamath**

If you found this project useful, feel free to star the repository.
