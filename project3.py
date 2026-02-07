# Importing All Requird Librery
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the DataSet
try: 
    df = pd.read_csv('Airline_Flight_Data_Cleaned.csv')
except FileNotFoundError:
    print('CSV File not found. Please check the file path.')
    exit()

#Display the First Few Rows of the DataFrame
print(df.head())

# Display the Shape of the DataFrame
print(df.shape)
print(df.info())

# Show The name OF columns
print(df.columns)

# Drop Unnecessary Columns
if 'days_left' in df.columns:
    df.drop(['days_left'], axis=1, inplace=True)

# Desplay Data Types
print(df.dtypes)
print(df.describe())

def check_missing_duplicates(df):
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())


# Drop Duplicate Rows
df.drop_duplicates(inplace=True)
# Verify Duplicates are Removed
print(df.duplicated().sum())

df.to_csv('Airline_Flight_Data_Cleaned.csv', index=False)

metadata = {
    "initial_shape": (300155,11),
    "final_shape": df.shape,
    "removed_columns": [ 'days_left'],
    "missing_values_filled": df.isnull().sum().to_dict()
}
print(metadata)

# Average Price by Airline
avg_price_airline = df.groupby("airline")["price"].mean().sort_values()
print(avg_price_airline)


df.groupby(["source_city", "destination_city"])["price"].mean().sort_values()
df.groupby(["source_city", "destination_city"]).size().sort_values(ascending=False)
df.groupby("stops")["price"].mean()

print(df)

# Visualizations
plt.figure(figsize=(10,6))
sns.barplot(x=df.groupby("airline")["price"].mean().index, y=df.groupby("airline")["price"].mean().values)
plt.title("Average Price by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.show()

avg_price_airline = df.groupby("airline")["price"].mean().sort_values()
print("Insight: The cheapest airline on average is:", avg_price_airline.index[0])
print("Insight: The most expensive airline on average is:", avg_price_airline.index[-1])


# Compair OF Economy and Business Class Prices
plt.figure(figsize=(6,5))
sns.barplot(x="class", y="price", data=df, estimator=np.mean, ci=None)
plt.title("Economy vs Business Ticket Prices")
plt.show()

# Top 10 Busiest Routes
top_routes = df.groupby(["source_city","destination_city"]).size().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
top_routes.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Top 10 Busiest Routes")
plt.ylabel("Number of Flights")
plt.show()

# Price Distribution by Number of Stops
plt.figure(figsize=(12,6))
sns.barplot(x="stops", y="price", hue="airline", data=df, estimator=np.mean, ci=None)
plt.title("Average Price by Stops and Airline")
plt.show()

# Top 10 expensive routes
expensive_routes = df.groupby(["source_city", "destination_city"])["price"].mean().sort_values(ascending=False).head(10)
print(expensive_routes)

plt.figure(figsize=(8,6))
sns.boxplot(x="airline", y="price", data=df)
plt.xticks(rotation=45)
plt.title("Price Outliers by Airline")
plt.show()


# Sklearn Model Implementation
# Copy DataFrame for modeling
df_model = df.copy()

# Encode categorical features properly
label_cols = ["airline", "source_city", "destination_city", "stops", "class"]
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le  # Save encoder for each column

print(df_model.head())

# Feature selection
X = df_model[["airline", "source_city", "destination_city", "stops", "class", "duration"]]
y = df_model["price"]

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

# Correlation heatmap
#plt.figure(figsize=(8,6))
'''sns.heatmap(df_model.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()'''

# ---------- FIXED FUNCTION ----------
def predict_price(airline, source_city, destination_city, stops, travel_class, duration):
    sample = pd.DataFrame({
        "airline": [airline],
        "source_city": [source_city],
        "destination_city": [destination_city],
        "stops": [stops],
        "class": [travel_class],
        "duration": [duration]
    })
    
    # Encode sample using trained encoders
    for col in label_cols:
        le = label_encoders[col]
        # Check if the category exists in training labels
        if sample[col].iloc[0] in le.classes_:
            sample[col] = le.transform(sample[col])
        else:
            # If unseen label, assign a default encoding (0)
            print(f"⚠️ Warning: '{sample[col].iloc[0]}' not seen in training data for '{col}'")
            sample[col] = [0]
    
    predicted_price = model.predict(sample)
    return predicted_price[0]

# Example Prediction
predicted_price = predict_price("Air India", "Delhi", "Mumbai", "non-stop", "Economy", 2.5)
print("\nPredicted Price for given flight:", round(predicted_price, 2))

# Summary of Model Performance
print("\nModel Summary:")
print(f"Mean Absolute Error (MAE): {round(mae, 2)}")
print(f"Root Mean Squared Error (RMSE): {round(rmse, 2)}")
print(f"R² Score: {round(r2, 4)}")

# launching PowerBI after Processing

pbix_file_path = r"C:\Users\ankur\OneDrive\Documents\New python\Project3.pbix" 

if os.path.exists(pbix_file_path):
    try:
        subprocess.run(["start", "Power BI", pbix_file_path], shell=True)
        print(f"Power BI file '{pbix_file_path}' opened successfully!")
    except Exception as e:
        print(f"Error launching Power BI: {e}")
else:
    print(f"Power BI file not found at {pbix_file_path}")