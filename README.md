# Customer Churn Prediction - ML Project

-----

## 🚀 Project Overview

This project aims to build a machine learning model capable of predicting customer churn (i.e., whether a customer will cancel their service or subscription). It covers key stages of a typical Machine Learning lifecycle, starting from synthetic data generation and comprehensive data preprocessing.

The goal is to demonstrate practical application of data science and machine learning techniques to a common business problem.

-----

## 🎯 Project Goal

To develop a robust predictive model that identifies customers at high risk of churn, enabling businesses to take proactive measures for retention.

-----

## 📂 Project Structure

```
customer_churn_prediction/
├── data/
│   └── customer_churn_data.csv  # Generated synthetic customer data
├── models/
│   └── preprocessor.pkl         # Saved data preprocessor (from scikit-learn)
│   └── (churn_model.pkl will go here)
├── src/
│   ├── data_generator.py        # Script to generate synthetic customer data
│   ├── preprocessing.py         # Script for data cleaning and transformation
│   └── __init__.py              # Makes 'src' a Python package
├── main.py                      # Orchestrates the project via CLI (coming soon)
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

-----

## ⚙️ Key Features Implemented (So Far)

  * **Synthetic Data Generation**: Creates a realistic, yet synthetic, customer dataset with various features (demographics, services, charges) and a 'Churn' target variable. This allows for development without sensitive real-world data.
  * **Comprehensive Data Preprocessing**: Handles common data preparation tasks including:
      * Missing value imputation (specifically for `TotalCharges`).
      * Feature-target separation.
      * Categorical feature encoding (One-Hot Encoding).
      * Numerical feature scaling (StandardScaler).
      * Data splitting into training and testing sets, ensuring stratified sampling for balanced classes.
      * Saving the fitted preprocessor for consistent future predictions.

-----

## 🛠️ Tech Stack

  * **Python (3.9+)**
  * **NumPy**: For numerical operations.
  * **Pandas**: For data manipulation and analysis.
  * **Scikit-learn**: For preprocessing utilities and (later) machine learning algorithms.
  * **Matplotlib / Seaborn**: For data visualization (used implicitly in EDA, not directly in current scripts but available).
  * **`joblib`**: For saving/loading `scikit-learn` objects.

-----

## 📦 How to Run (Current Stage)

Follow these steps to generate and preprocess the data for the project.

1.  **Clone the Repository (if you haven't already):**

    ```bash
    git clone https://github.com/AakashChettay/customer-churn-prediction.git
    cd customer_churn_prediction
    ```

2.  **Install Dependencies:**
    Navigate to the project root directory (`customer_churn_prediction/`) and install all required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Generate Synthetic Data:**
    Run the `data_generator.py` script. This will create the `customer_churn_data.csv` file in the `data/` directory.

    ```bash
    python src/data_generator.py
    ```

    *You should see a confirmation message like "Synthetic data saved to 'data/customer\_churn\_data.csv'".*

4.  **Perform Data Preprocessing:**
    Run the `preprocessing.py` script. This will load the generated data, preprocess it, and save the fitted `preprocessor.pkl` model to the `models/` directory.

    ```bash
    python src/preprocessing.py
    ```

    *You should see logs detailing the preprocessing steps and a confirmation like "Preprocessor saved to 'models/preprocessor.pkl'".*

-----

## ✅ Checkpoints & Evaluation (Current Stage)

This section evaluates the functionalities implemented so far.

### **1. Synthetic Data Generation**

  * **Status:** **PASS**
  * **Evaluation:** The `data_generator.py` script successfully creates a CSV file (`customer_churn_data.csv`) with the specified number of samples and relevant features, including a 'Churn' target variable.
  * **Implementation Detail:** Uses `pandas` and `numpy` to create a DataFrame with randomized but correlated features and target.

### **2. Data Loading**

  * **Status:** **PASS**
  * **Evaluation:** The preprocessing script successfully loads the generated CSV data into a Pandas DataFrame.
  * **Implementation Detail:** Uses `pd.read_csv()`.

### **3. Missing Value Handling**

  * **Status:** **PASS**
  * **Evaluation:** The preprocessing pipeline correctly identifies and handles potential missing values (specifically empty strings in `TotalCharges`), converting them to numeric.
  * **Implementation Detail:** `df['TotalCharges'].replace(' ', np.nan)` followed by `fillna(0)`.

### **4. Feature-Target Separation**

  * **Status:** **PASS**
  * **Evaluation:** The data is correctly split into features (`X`) and the target variable (`y`, 'Churn').
  * **Implementation Detail:** `df.drop('Churn', axis=1)` and `df['Churn']`.

### **5. Categorical Feature Encoding**

  * **Status:** **PASS**
  * **Evaluation:** Categorical features are correctly identified and transformed into numerical format using One-Hot Encoding, making them suitable for ML models.
  * **Implementation Detail:** `OneHotEncoder` within `ColumnTransformer`.

### **6. Numerical Feature Scaling**

  * **Status:** **PASS**
  * **Evaluation:** Numerical features are scaled (standardized) to have a mean of 0 and standard deviation of 1, which is crucial for many ML algorithms.
  * **Implementation Detail:** `StandardScaler` within `ColumnTransformer`.

### **7. Data Splitting (Train/Test)**

  * **Status:** **PASS**
  * **Evaluation:** The data is split into training and testing sets (80/20 ratio by default), and `stratify=y` ensures that the proportion of churners is maintained in both sets.
  * **Implementation Detail:** `sklearn.model_selection.train_test_split` with `stratify`.

### **8. Preprocessor Persistence**

  * **Status:** **PASS**
  * **Evaluation:** The fitted preprocessing pipeline is saved to a `.pkl` file, allowing it to be reused for consistent transformation of new, unseen data during prediction.
  * **Implementation Detail:** `joblib.dump(preprocessor, PREPROCESSOR_FILE)`.

-----

## 🔜 Next Steps

The next phase of this project will involve:

  * **Model Training**: Implementing and training various classification models (e.g., Logistic Regression, Random Forest, Gradient Boosting).
  * **Model Evaluation**: Rigorously evaluating model performance using appropriate metrics.
  * **Model Saving**: Persisting the trained model for future use.
  * **Prediction**: Creating functionality to make predictions on new data.
  * **CLI Orchestration**: Building `main.py` to tie all these steps together via command-line arguments.

-----