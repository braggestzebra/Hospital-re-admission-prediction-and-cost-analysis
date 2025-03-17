import pandas as pd

# Load dataset
file_path = "data/FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv"
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())

# Check for missing values
print("\nMissing values in each column:\n", df.isnull().sum())
# Drop unnecessary columns
df = df.drop(columns=["Facility Name", "Facility ID", "Footnote", "Start Date", "End Date"])

# Convert "Too Few to Report" to NaN in 'Number of Readmissions'
df["Number of Readmissions"] = pd.to_numeric(df["Number of Readmissions"], errors='coerce')

# Fill missing values in numeric columns with the median
numeric_cols = ["Number of Discharges", "Predicted Readmission Rate", "Expected Readmission Rate", "Number of Readmissions"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


# Display the remaining columns
print("Remaining columns after dropping unnecessary ones:\n", df.columns)
# Drop rows where the target variable ('Excess Readmission Ratio') is missing
df = df.dropna(subset=["Excess Readmission Ratio"])

# Fill missing values in numeric columns with the median
numeric_cols = ["Number of Discharges", "Predicted Readmission Rate", "Expected Readmission Rate", "Number of Readmissions"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Convert categorical columns (State, Measure Name) into numerical using one-hot encoding
df = pd.get_dummies(df, columns=["State", "Measure Name"], drop_first=True)

# Check if there are any missing values left
print("\nMissing values after cleaning:\n", df.isnull().sum())

# Save cleaned dataset
df.to_csv("data/cleaned_hospital_data.csv", index=False)
print("\n✅ Data cleaning complete! Cleaned data saved.")
# Load cleaned dataset and display basic stats
df_cleaned = pd.read_csv("data/cleaned_hospital_data.csv")
print(df_cleaned.info())  # Check data types
print(df_cleaned.head())  # Show first rows
