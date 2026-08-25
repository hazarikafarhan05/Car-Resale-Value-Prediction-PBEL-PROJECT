import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("../data/Used_Car_Price_Prediction.csv")

df = df.drop_duplicates()

numeric_columns = [
    "yr_mfr",
    "kms_run",
    "times_viewed",
    "total_owners",
    "broker_quote",
    "original_price",
    "emi_starts_from",
    "booking_down_pymnt"
]

categorical_columns = [
    "car_name",
    "fuel_type",
    "city",
    "body_type",
    "transmission",
    "variant",
    "registered_city",
    "registered_state",
    "rto",
    "source",
    "make",
    "model",
    "car_availability",
    "car_rating",
    "fitness_certificate"
]

boolean_columns = [
    "assured_buy",
    "is_hot",
    "reserved",
    "warranty_avail"
]

features = numeric_columns + categorical_columns + boolean_columns

X = df[features]
y = df["sale_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numeric_transformer = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
]

categorical_transformer = [
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_columns),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_columns
        ),
        ("bool", SimpleImputer(strategy="most_frequent"), boolean_columns)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = GradientBoostingRegressor(
    random_state=42
)

model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Gradient Boosting Model Performance")
print("-----------------------------------")
print(f"MAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R²   : {r2:.4f}")
