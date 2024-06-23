import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# LOADING DATA AND DEFINING INDEPENDENT AND DEPENDENT VARIABLES

# Load the susm [mm/month], temperature [degrees Celcius], and precipitation [mm/month] data
file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\susm_temperature_precipitation_historical_Brazil.csv'
data = pd.read_csv(file_path)

# Define independent (X) and dependent (y) variables
y = data['avg_susm'] # avg_susm in [mm/month]
X = data[['avg_temp', 'precipitation']] # avg_temp in [degrees Celcius] and precipitation in [mm/month]

# CHECK FOR MULTICOLLINEARITY

# Calculate Variance Inflation Factor (VIF) for each feature to check for multicollinearity (assuming 'VIF' < 10 as threshold)
# Multicollinearity occurs when independent variables are highly correlated, which can distort the regression model and lead to unreliable coefficient estimates.
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("______________________________________________________________________")
print(vif_data)
# VIF value for the independent variables should be checked, ensure they are < 10

# CREATE MULTIPLE LINEAR REGRESSION MODEL AND CHECK SIGNIFICANCE

# Build the linear regression model (without the constant term)
model = sm.OLS(y, X).fit()

# Print the linear regression model summary and check significance (assuming 'Prob (F-statistic' and 'P>|t|' < 0.05 as threshold)
print("______________________________________________________________________")
print(model.summary())
# Check the overall model significance and the coefficients' significance

# Print the linear regression model formula 
coeff_avg_temp = round(model.params['avg_temp'], 9)
coeff_precipitation = round(model.params['precipitation'], 9)
formula = f"avg_susm = {coeff_avg_temp} * avg_temp + {coeff_precipitation} * precipitation" # avg_susm in [mm/month]
print("______________________________________________________________________")
print("Linear Regression Model Formula (through origin):")
print(formula)
print("______________________________________________________________________")

# Predict soil moisture [mm/month] based on the input values of the independent variables 'X' using the regression model
y_pred = model.predict(X)

# Ensure no negative predictions (since soil moisture cannot be negative)
y_pred = y_pred.clip(lower=0)

# VISUALIZATION

# Visualization: 3D Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Labels and title
ax.set_xlabel('Precipitation [mm/month]')
ax.set_ylabel('Temperature [degrees Celcius]')
ax.set_zlabel('Soil Moisture [mm/month]')
plt.title('3D Scatter Plot of Soil Moisture, Temperature, and Precipitation')
# Plot graph
ax.scatter(data['precipitation'], data['avg_temp'], data['avg_susm'], c='r', marker='o')
plt.show()

# Visualization: Actual vs Predicted Soil Moisture
plt.scatter(y, y_pred)
# Labels and title
plt.xlabel('Actual Soil Moisture [mm/month]')
plt.ylabel('Predicted Soil Moisture [mm/month]')
plt.title('Actual vs Predicted Soil Moisture')
# Plot graph
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

# Residual Analysis: Residuals vs Predicted
residuals = y - y_pred
plt.scatter(y_pred, residuals)
# Labels and title
plt.xlabel('Predicted Soil Moisture [mm/month]')
plt.ylabel('Residuals [mm/month]')
plt.title('Residuals vs Predicted Soil Moisture')
# Plot graph
plt.axhline(y=0, color='r', linestyle='--')
plt.show()




# CALCULATE CHANGE IN STORAGE PER STATE WITH THE MULTIPLE LINEAR REGRESSION MODEL [m^3]

# File paths and area for each state
base_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python'
files = [
    {
        "input": f"{base_path}\Precipitation_temperature_Minas_Gerais.csv",
        "output": f"{base_path}\\total_change_in_storage\Precipitation_temperature_Minas_Gerais_totChangeInStorage.csv",
        "area": 269481764694.99707 # Area Minas Gerais in m^2
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Bahia.csv",
        "output": f"{base_path}\\total_change_in_storage\Precipitation_temperature_Bahia_totChangeInStorage.csv",
        "area": 298199130706.521 # Area Bahia in m^2
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Pernambuco.csv",
        "output": f"{base_path}\\total_change_in_storage\Precipitation_temperature_Pernambuco_totChangeInStorage.csv",
        "area": 61848813793.99841 # Area Pernambuco in m^2
    }
]

# Process each file
for file_info in files:
    # Read precipitation and temperature data from the file
    data = pd.read_csv(file_info["input"])

    # Calculate avg_temp [degrees Celcius]
    data['avg_temp'] = (data['Tmin'] + data['Tmax']) / 2 

    # Calculate avg_susm using the linear regression model formula in [mm]
    data['avg_susm'] =  coeff_avg_temp * data['avg_temp'] + coeff_precipitation * data['precipitation']

    # Calculate total_susm in [m^3/month]
    data['total_susm'] = (data['avg_susm'] / 1000) * file_info["area"]

    # Calculate total change in storage by determining difference between a total susm and the total susm of the month before [mm/month]
    data['total_change_in_storage'] = data['total_susm'].diff().fillna(0)

    # Set change of storage to 0 at the start of each new SSP
    rows_to_reset = [0, 12, 60, 108, 156]
    data.loc[rows_to_reset, 'total_change_in_storage'] = 0

    # Save the processed data to a new CSV file
    data.to_csv(file_info["output"], index=False)

    print(f"Processed file saved to: {file_info['output']}")



# CALCULATE CHANGE IN STORAGE FOR ENTIRE BASIN [m^3]

# File paths of the processed files
file_paths = [
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\Precipitation_temperature_Minas_Gerais_totChangeInStorage.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\Precipitation_temperature_Bahia_totChangeInStorage.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\Precipitation_temperature_Pernambuco_totChangeInStorage.csv'
]

# Read the data from each file into a list of dataframes
dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Extract the required columns from the first dataframe (assuming all have the same year, month, ssp columns)
result_df = dataframes[0][['year', 'month', 'ssp']].copy()

# Initialize the total_change_in_storage column in the result dataframe to zero
result_df['total_change_in_storage'] = 0

# Sum the total_change_in_storage columns across all dataframes
for df in dataframes:
    result_df['total_change_in_storage'] += df['total_change_in_storage']

# Save the result to a new CSV file
output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\total_change_in_storage_all_states.csv'
result_df.to_csv(output_file_path, index=False)

print(f"Summed total_change_in_storage saved to: {output_file_path}")




