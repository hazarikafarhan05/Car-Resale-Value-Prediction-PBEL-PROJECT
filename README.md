Project Overview:

This project focuses on predicting the resale/sale price of used cars using Machine Learning and Deep Learning techniques.

The model learns from various car-related features such as manufacturing year, kilometers driven, fuel type, location, transmission, number of previous owners, broker quote, original price, and other vehicle details.

The project compares multiple regression models and selects the best-performing model based on evaluation metrics.

---

Problem Statement:

Estimating the resale value of a used car can be difficult because its price depends on several factors such as age, mileage, brand, model, condition, location, and market demand.

The objective of this project is to build a regression system that can predict the selling price of a used car based on its available features.

---

Objective:

- Analyze used-car market data through Exploratory Data Analysis (EDA).
- Identify important factors affecting used-car prices.
- Preprocess numerical and categorical features.
- Train multiple Machine Learning and Deep Learning regression models.
- Compare models using MAE, RMSE, and R² Score.
- Select the best-performing model for predicting resale prices.
- Test the selected model with new car details.

---

Dataset:

The dataset contains 7,400 used-car listings with 29 features from the Indian used-car market. The target variable is "sale_price".

Dataset Source

Kaggle – Used Car Price Prediction

"https://www.kaggle.com/datasets/vrajesh0sharma7/used-car-price-prediction" (https://reference-url-citation.invalid/2)

Target Variable:

"- sale_price"

Important Features:

- "car_name"
- "yr_mfr"
- "fuel_type"
- "kms_run"
- "city"
- "times_viewed"
- "body_type"
- "transmission"
- "variant"
- "make"
- "model"
- "total_owners"
- "broker_quote"
- "original_price"
- "car_rating"
- "fitness_certificate"
- "warranty_avail"

---

Exploratory Data Analysis:

The following aspects were explored:

- Number of vehicles, manufacturers, and models
- Distribution of fuel types
- Transmission types
- Body types
- City-wise distribution
- Distribution of used-car sale prices
- Relationship between manufacturing year and sale price
- Relationship between kilometers driven and sale price
- Average sale price by fuel type
- Sale price based on number of previous owners

Key Dataset Insights:

- 7,400 original vehicle records
- 27 manufacturers
- 185 car models
- Petrol cars form the largest fuel category.
- Manual transmission cars are more common than automatic cars.
- Hatchbacks are the most common body type.
- Mumbai, Bengaluru, and New Delhi have a large number of listings.

---

Data Preprocessing:

The following preprocessing steps were performed:

1. Removed duplicate records.
2. Converted relevant numerical columns to numeric data types.
3. Removed columns that were considered unnecessary for prediction.
4. Separated features ("X") and target variable ("y").
5. Identified numerical and categorical features.
6. Split the dataset into training and testing sets using an 80:20 ratio.
7. Filled missing numerical values using median imputation.
8. Filled missing categorical values using most-frequent imputation.
9. Applied One-Hot Encoding to categorical features.
10. Applied Standard Scaling to numerical features.
11. Combined the processed numerical and categorical features for model training.

---

Models Implemented:

Four regression models were trained and evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. Artificial Neural Network (ANN)

---

Model Performance:

The models were evaluated using:

- MAE (Mean Absolute Error) – lower is better
- RMSE (Root Mean Squared Error) – lower is better
- R² Score – higher is better

Model| MAE| RMSE| R² Score
Linear Regression| ₹26,157.17| ₹86,589.15| 0.9168
Random Forest| ₹18,883.84| ₹86,717.33| 0.9166
Gradient Boosting| ₹18,915.29| ₹80,672.97| 0.9278
ANN| ₹25,371.52| ₹85,470.54| 0.9190

---

Best Model:

The Gradient Boosting Regressor achieved the best overall performance.

Best Model Results

- MAE: ₹18,915.29
- RMSE: ₹80,672.97
- R² Score: 0.9278

An R² score of approximately 0.93 indicates that the model explains a large proportion of the variation in used-car sale prices within the test dataset.

---

Visualizations:

The project includes visualizations for:

- Sale price distribution- <img width="700" height="382" alt="image" src="https://github.com/user-attachments/assets/5fabd6b7-280c-40a3-8e94-aafc8ed5b367" />

- Sale price vs manufacturing year- <img width="592" height="382" alt="image" src="https://github.com/user-attachments/assets/d39c01a9-9f6e-4d22-8053-a0c151644fde" />

- Sale price vs kilometers driven- <img width="592" height="382" alt="image" src="https://github.com/user-attachments/assets/c377e4a1-2d2a-4e60-bdb4-f505bf408344" />

- Average sale price by fuel type- <img width="598" height="382" alt="image" src="https://github.com/user-attachments/assets/fd3617fb-7153-4fb6-b77a-8b0a5c26578d" />

- MAE, RMSE, and R² model comparison- <img width="1789" height="490" alt="image" src="https://github.com/user-attachments/assets/205b50e5-4b2b-499d-906d-1c286feee82b" />



---

Prediction with New Input:

The project also allows the user to enter details of a new used car, including:

-Car Name: maruti swift
-Manufacturing Year: 2018
-Fuel Type: petrol
-Kilometers Driven: 35000
-City: mumbai
-Times Viewed: 1200
-Body Type: hatchback
-Transmission: manual
-Variant: vxi
-Assured Buy (True/False): true
-Registered City: mumbai
-Registered State: maharashtra
-Is Hot (True/False): false
-Manufacturer: maruti
-Model: swift
-Total Owners: 1
-Broker Quote (₹): 450000
-Original Price (₹): 600000
-Car Rating: great
-Fitness Certificate (True/False): true
-Warranty Available (True/False): false
The processed input is passed to the Gradient Boosting model, which produces an estimated resale price.

Example Output

Estimated Resale Price: ₹478,853.38

---

Technologies Used:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook / Google Colab

---

How to Run:

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Used-Car-Resale-Value-Prediction

2. Install dependencies

pip install -r requirements.txt

3. Open the notebook

Open:

notebooks/Car_Resale_Value_project_using_ML_and_DL.ipynb

The notebook can be executed using Google Colab or Jupyter Notebook.

4. Load the dataset

Download the dataset from Kaggle and place it inside the "data/" folder.

---

Future Improvements:

1. Model Improvement: Explore advanced ML/DL models and hyperparameter tuning to improve prediction accuracy.
2. Real-Time Price Prediction: Develop a user-friendly application that provides instant resale price estimates based on car details.
3. More Data Integration: Include real-time market trends, location-based pricing, and updated vehicle listings for better predictions. 


---

Conclusion:

This project demonstrates how Machine Learning and Deep Learning can be used to estimate the resale value of used cars.

Among the tested models, Gradient Boosting achieved the highest R² score of 0.9278 and the lowest RMSE, making it the best-performing model for this project.

The project provides a complete workflow from data analysis and preprocessing to model training, evaluation, comparison, and real-world price prediction.
