# --- Imports ---
import pandas as pd # Import the pandas library, commonly aliased as 'pd'
import numpy as np  # Import the numpy library, commonly aliased as 'np'
import os           # Import the os module for interacting with the operating system (e.g., file paths)
import logging      # Import the logging module for structured logging output

# Configure logging for this specific script.
# This sets up how messages (INFO, WARNING, ERROR) will be displayed in the console.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Function Definition: generate_customer_churn_data ---
def generate_customer_churn_data(num_samples: int = 1000, output_path: str = None):
    """
    Generates a synthetic (fake) dataset for customer churn prediction.
    This function creates realistic-looking data without using real customer information.

    Args:
        num_samples (int): The number of customer records (rows) to generate in the dataset.
                           Default is 1000.
        output_path (str): The full path where the generated CSV file should be saved.
                           If None, it defaults to a path relative to the script.
    """
    # Log an informational message indicating the start of data generation.
    logging.info(f"Generating {num_samples} synthetic customer churn data samples...")

    # --- Define Features (Columns) and Generate Data for Each ---
    # We use numpy's random functions to create diverse data.

    # 'gender': Randomly choose 'Male' or 'Female' for each sample.
    gender = np.random.choice(['Male', 'Female'], num_samples)

    # 'SeniorCitizen': Randomly choose 0 (No) or 1 (Yes).
    # 'p=[0.8, 0.2]' means 80% will be 0, 20% will be 1.
    senior_citizen = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])

    # 'Partner': Whether the customer has a partner ('Yes' or 'No').
    partner = np.random.choice(['Yes', 'No'], num_samples)

    # 'Dependents': Whether the customer has dependents ('Yes' or 'No').
    dependents = np.random.choice(['Yes', 'No'], num_samples)

    # 'tenure': How many months the customer has been with the company.
    # Random integer between 1 and 71 (inclusive of 1, exclusive of 72).
    tenure = np.random.randint(1, 72, num_samples)

    # 'PhoneService': Whether the customer has phone service ('Yes' or 'No').
    phone_service = np.random.choice(['Yes', 'No'], num_samples)

    # 'MultipleLines': Whether they have multiple phone lines.
    # 'No phone service' is an option if 'PhoneService' is 'No'.
    multiple_lines = np.empty(num_samples, dtype='U16') # U16 for string length

    for i in range(num_samples):
        if phone_service[i] == 'No':
            multiple_lines[i] = 'No phone service'
        else:
            # If they have phone service, then they can either have multiple lines or not
            multiple_lines[i] = np.random.choice(['No', 'Yes'])

    # 'InternetService': Type of internet service.
    # 'p=[0.4, 0.4, 0.2]' means 40% DSL, 40% Fiber optic, 20% No internet.
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], num_samples, p=[0.4, 0.4, 0.2])

    # Various service features: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
    # These can be 'No', 'Yes', or 'No internet service' (if InternetService is 'No').
    online_security = np.random.choice(['No', 'Yes', 'No internet service'], num_samples)
    online_backup = np.random.choice(['No', 'Yes', 'No internet service'], num_samples)
    device_protection = np.random.choice(['No', 'Yes', 'No internet service'], num_samples)
    tech_support = np.random.choice(['No', 'Yes', 'No internet service'], num_samples)
    streaming_tv = np.random.choice(['No', 'Yes', 'No internet service'], num_samples)
    streaming_movies = np.random.choice(['No', 'Yes', 'No internet service'], num_samples)

    # 'Contract': Type of contract.
    # 'p=[0.6, 0.2, 0.2]' means 60% month-to-month, 20% one year, 20% two year.
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples, p=[0.6, 0.2, 0.2])

    # 'PaperlessBilling': Whether they receive paperless bills.
    paperless_billing = np.random.choice(['Yes', 'No'], num_samples)

    # 'PaymentMethod': How the customer pays.
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], num_samples)

    # 'MonthlyCharges': Simulated monthly bill.
    # Start with a base uniform distribution.
    monthly_charges = np.random.uniform(20, 120, num_samples)
    # Add some correlation: Fiber optic internet usually means higher charges.
    # We select only the samples where internet_service is 'Fiber optic' and add to their charges.
    monthly_charges[internet_service == 'Fiber optic'] += np.random.uniform(10, 30, (internet_service == 'Fiber optic').sum())
    # No internet service usually means lower charges.
    monthly_charges[internet_service == 'No'] -= np.random.uniform(10, 20, (internet_service == 'No').sum())
    # Round to 2 decimal places.
    monthly_charges = np.round(monthly_charges, 2)

    # 'TotalCharges': Simulated total charges over tenure.
    # Base calculation: monthly_charges * tenure.
    total_charges = monthly_charges * tenure
    # Add some random noise to make it more realistic.
    total_charges += np.random.normal(0, 50, num_samples)
    # Ensure no negative total charges (can happen with random noise).
    total_charges[total_charges < 0] = 0
    # Round to 2 decimal places.
    total_charges = np.round(total_charges, 2)

    # --- Target Variable: 'Churn' (0 = No Churn, 1 = Churn) ---
    # We'll simulate churn based on some common patterns observed in real data.
    churn = np.zeros(num_samples, dtype=int) # Initialize all churn values to 0 (No Churn).
    churn_prob = np.zeros(num_samples) # Initialize an array to store churn probabilities for each customer.

    # Increase churn probability for certain conditions:
    # Customers on month-to-month contracts are more likely to churn.
    churn_prob[contract == 'Month-to-month'] += 0.3
    # Customers with low tenure (new customers) are sometimes more likely to churn.
    churn_prob[tenure < 12] += 0.2
    # Customers with very high monthly charges might be unhappy and churn.
    churn_prob[monthly_charges > 80] += 0.15
    # Customers without tech support are more likely to churn.
    churn_prob[tech_support == 'No'] += 0.2

    # Decrease churn probability for certain conditions:
    # Customers on 2-year contracts are less likely to churn.
    churn_prob[contract == 'Two year'] -= 0.3

    # Finally, assign churn (1) or no churn (0) based on a random draw
    # against the calculated churn_prob.
    churn = (np.random.rand(num_samples) < churn_prob).astype(int)

    # --- Create Pandas DataFrame ---
    # Combine all the generated arrays into a single Pandas DataFrame.
    data = pd.DataFrame({
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn # This is our target variable
    })

    # --- Save DataFrame to CSV ---
    # Determine the output directory.
    # If output_path is not provided, construct a default path relative to the script's location.
    if output_path is None:
        # os.path.dirname(__file__) gets the directory of the current script (src/).
        # '..' moves one level up (to customer_churn_prediction/).
        # 'data' then goes into the data/ directory.
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'customer_churn_data.csv')

    # Create the output directory if it doesn't exist.
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True) # exist_ok=True prevents error if dir already exists.

    # Save the DataFrame to a CSV file.
    # index=False prevents pandas from writing the DataFrame index as a column in the CSV.
    data.to_csv(output_path, index=False)
    logging.info(f"Synthetic data saved to '{output_path}'")

# --- Conditional Execution Block ---
# This block ensures that generate_customer_churn_data() is called only when
# this script is executed directly (e.g., `python src/data_generator.py`),
# not when it's imported as a module into another script.
if __name__ == "__main__":
    # Call the function to generate data.
    # We explicitly pass the output_path here to ensure it goes into the 'data' folder
    # relative to the project root, even if this script is run from 'src/'.
    generate_customer_churn_data(num_samples=2000, output_path=os.path.join(os.path.dirname(__file__), '..', 'data', 'customer_churn_data.csv'))