import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

# Ensure 'model/' directory exists
os.makedirs("model", exist_ok=True)

# Save the trained model



#loading the data to the code
df = pd.read_csv('data/cleaned_hospital_data.csv')

print(df.columns)  # Print columns to check the correct column name
X = df.drop(columns=["Excess Readmission Ratio"])
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy/indicator variables
y = df["Excess Readmission Ratio"]
# defining the model features [x] and target [y]

X = X.fillna(X.median())  # Fill missing values with median

y = y.loc[X.index]  # Align the target variable with the features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split into training and testing data

#training model
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
# Evaluating the model

#saving the model for later use
pickle.dump(model, open("model/readmission_model.pkl", "wb"))
print("✅ Model has been saved successfully!")
