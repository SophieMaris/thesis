import math
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Define month-to-number conversion
month_to_num = {
    " January": 1, " February": 2, " March": 3, " April": 4, " May": 5,
    " June": 6, " July": 7, " August": 8, " September": 9, " October": 10,
    " November": 11, " December": 12
}

# FUNCTIONS
# Function to calculate extraterrestrial radiation (Ra) and intermediate values
def calculate_ra(month, latitude_rad):
    # Convert month to number
    month_num = month_to_num[month]
    
    # Calculate day of the year for the 15th of the month
    day_of_year = 30.42 * month_num - 15.23
    
    # Calculate the relative distance (d_r)
    d_r = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
    
    # Calculate the declination (delta)
    delta = 0.4093 * math.sin((2 * math.pi * day_of_year / 365) - 1.405)
    
    # Calculate the sunset hour angle (omega_s)
    omega_s = math.acos(-math.tan(latitude_rad) * math.tan(delta))
    
    # Calculate Ra using the provided formula [mm/day]
    Ra = 15.392 * d_r * (omega_s * math.sin(latitude_rad) * math.sin(delta) +
                         math.cos(latitude_rad) * math.cos(delta) * math.sin(omega_s))
    return day_of_year, d_r, delta, omega_s, Ra

# Function to calculate PET in [mm/month]
def calculate_pet(T_min, T_max, Ra, month_to_days):
    T_mean = (T_max + T_min) / 2
    PET = 0.0023 * Ra * (T_mean + 17.8) * math.sqrt(T_max - T_min) * month_to_days
    return PET

# Function to calculate max_AET using the coefficients from the regression model
def calculate_max_aet(T_min, T_max, precipitation, coeff_avg_temp, coeff_precipitation):
    T_mean = (T_max + T_min) / 2
    max_AET = coeff_avg_temp * T_mean + coeff_precipitation * precipitation
    return max_AET

# Function to perform multiple linear regression and visualize results
def perform_multiple_regression(file_path):
    # Load data
    data = pd.read_csv(file_path)
    
    # Define independent (X) and dependent (y) variables
    y = data['avg_ssm']  # avg_susm in [mm/month]
    X = data[['avg_temp', 'precipitation']]  # avg_temp in [degrees Celcius] and precipitation in [mm/month]
    
    # CHECK FOR MULTICOLLINEARITY
    # Calculate Variance Inflation Factor (VIF) for each feature
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("______________________________________________________________________")
    print(vif_data)
    # VIF value for the independent variables (1.078178) < 10, so there is no multicollinearity
    
    # CREATE MULTIPLE LINEAR REGRESSION MODEL AND CHECK SIGNIFICANCE
    # Build the linear regression model
    model = sm.OLS(y, X, hasconst=False).fit()
    
    # Print model summary
    print("______________________________________________________________________")
    print(model.summary())
    # Print linear regression model formula
    coeff_avg_temp = round(model.params['avg_temp'], 9)
    coeff_precipitation = round(model.params['precipitation'], 9)
    formula = f"avg_ssm = {coeff_avg_temp} * avg_temp + {coeff_precipitation} * precipitation"
    print("______________________________________________________________________")
    print("Linear Regression Model Formula:")
    print(formula)
    print("______________________________________________________________________")
    
    # VISUALIZATION
    # Visualization: 3D Scatter Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Precipitation [mm/month]')
    ax.set_ylabel('Temperature [degrees Celcius]')
    ax.set_zlabel('Soil Moisture [mm/month]')
    plt.title('3D Scatter Plot of Soil Moisture, Temperature, and Precipitation')
    ax.scatter(data['precipitation'], data['avg_temp'], data['avg_ssm'], c='r', marker='o')
    plt.show()
    
    # Visualization: Actual vs Predicted Soil Moisture
    y_pred = model.predict(X)
    plt.scatter(y, y_pred)
    plt.xlabel('Actual Soil Moisture [mm/month]')
    plt.ylabel('Predicted Soil Moisture [mm/month]')
    plt.title('Actual vs Predicted Soil Moisture')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.show()
    
    # Residual Analysis: Residuals vs Predicted
    residuals = y - y_pred
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Soil Moisture [mm/month]')
    plt.ylabel('Residuals [!!!!]')
    plt.title('Residuals vs Predicted Soil Moisture')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
    
    # Return coefficients for max_AET calculation
    return coeff_avg_temp, coeff_precipitation

# Function to calculate total AET in [m^3/month]
def calculate_total_aet(AET, area):
    return area * (AET / 1000) 

# Function to calculate beta using the given formula
def calculate_beta(St_prev, St):
    ratio = St_prev / St
    beta = (5 * ratio - 2 * ratio**2) / 3
    return beta, ratio

def process_state_data(input_file, output_file, latitude, area):
    # Convert latitude to radians
    latitude_rad = latitude * math.pi / 180
    
    # Read the input CSV file
    df = pd.read_csv(input_file)
       
    # Map month names to the number of days in each month
    month_to_days = {
        ' January': 31,
        ' February': 28,  # Consider leap years separately if needed
        ' March': 31,
        ' April': 30,
        ' May': 31,
        ' June': 30,
        ' July': 31,
        ' August': 31,
        ' September': 30,
        ' October': 31,
        ' November': 30,
        ' December': 31
    }
    
    # Calculate Ra, PET, max_AET, and intermediate values
    results = df.apply(lambda row: calculate_ra(row['month'], latitude_rad), axis=1)
    df[['day_of_month', 'd_r', 'delta', 'omega_s', 'Ra']] = pd.DataFrame(results.tolist(), index=df.index)
    df['PET'] = df.apply(lambda row: calculate_pet(row['Tmin'], row['Tmax'], row['Ra'], month_to_days[row['month']]), axis=1) # [mm/month]
    
    # Perform multiple linear regression and get coefficients
    file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\ssm_temperature_precipitation_historical_Brazil.csv'
    coeff_avg_temp, coeff_precipitation = perform_multiple_regression(file_path)
    
    # Calculate max_AET using the coefficients from regression model
    df['max_AET'] = df.apply(lambda row: calculate_max_aet(row['Tmin'], row['Tmax'], row['precipitation'], coeff_avg_temp, coeff_precipitation), axis=1)
    
    # Initialize previous month's max_AET
    St_prev = df['max_AET'].iloc[0]
    
    # Calculate AET, beta, total_AET, and ratio dynamically
    AET_list = []
    beta_list = []
    total_AET_list = []
    ratio_list = []
    AET_without_max_list = []
    for i, row in df.iterrows():
        St = row['max_AET']
        beta, ratio = calculate_beta(St_prev, St)
        
        # Calculate AET without max_AET constraint
        AET_without_max = row['PET'] * beta
        
        # Calculate AET with max_AET constraint
        AET = AET_without_max if AET_without_max <= row['max_AET'] else row['max_AET']
        
        total_AET = calculate_total_aet(AET, area)
        
        AET_list.append(AET)
        beta_list.append(beta)
        total_AET_list.append(total_AET)
        ratio_list.append(ratio)
        AET_without_max_list.append(AET_without_max)
        
        # Update St_prev for the next iteration
        St_prev = St
    
    # Add the calculated values to the dataframe
   


# Example usage (assuming you have a CSV input file and relevant parameters)
# process_state_data('input_file.csv', 'output_file.csv', latitude, area)

# File paths
base_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python'
files = [
    {
        "input": f"{base_path}\Precipitation_temperature_Minas_Gerais.csv",
        "output": f"{base_path}\\total_AET\Precipitation_temperature_Minas_Gerais_totAET.csv",
        "latitude": -17.8154808447, # Latitude Minas Gerais
        "area": 269481764694.99707 # Area Minas Gerais in m^2
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Bahia.csv",
        "output": f"{base_path}\\total_AET\Precipitation_temperature_Bahia_totAET.csv",
        "latitude": -11.7474248842, # Latitude Bahia
        "area": 298199130706.521 # Area Bahia in m^2
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Pernambuco.csv",
        "output": f"{base_path}\\total_AET\Precipitation_temperature_Pernambuco_totAET.csv",
        "latitude": -8.2945583277, # Latitude Pernambuco
        "area": 61848813793.99841 # Area Pernambuco
    }
]

# Process each file
for file in files:
    process_state_data(file['input'], file['output'], file['latitude'], file['area'])

# CALCULATE TOTAL AET FOR ENTIRE BASIN [m^3]

# File paths of the processed files
file_paths = [
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\Precipitation_temperature_Minas_Gerais_totAET.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\Precipitation_temperature_Bahia_totAET.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\Precipitation_temperature_Pernambuco_totAET.csv'
]

# Read the data from each file into a list of dataframes
dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Extract the required columns from the first dataframe (assuming all have the same year, month, ssp columns)
result_df = dataframes[0][['year', 'month', 'ssp']].copy()

# Initialize the total_AET column in the result dataframe to zero
result_df['total_AET'] = 0

# Sum the total_AET columns across all dataframes
for df in dataframes:
    result_df['total_AET'] += df['total_AET']

# Save the result to a new CSV file
output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\total_AET_all_states.csv'
result_df.to_csv(output_file_path, index=False)

print(f"Summed total_AET saved to: {output_file_path}")

import pandas as pd
# Assuming the dataframes and indices are already defined

input_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\Precipitation_temperature_Minas_Gerais_totAET.csv'
data = pd.read_csv(input_file_path)

indices = {
    '1995-2014': {
        'Historical 1995-2014': list(range(0,12)),
    },
    '2020-2039': {
        'Historical 1995-2014': list(range(0,12)),
        'SSP1': list(range(12, 24)),
        'SSP2': list(range(60, 72)),
        'SSP3': list(range(108, 120)),
        'SSP5': list(range(156, 168))
    },
    '2040-2059': {
        'Historical 1995-2014': list(range(0,12)),
        'SSP1': list(range(24, 36)),
        'SSP2': list(range(72, 84)),
        'SSP3': list(range(120, 132)),
        'SSP5': list(range(168, 180))
    },
    '2060-2079': {
        'Historical 1995-2014': list(range(0,12)),
        'SSP1': list(range(36, 48)),
        'SSP2': list(range(84, 96)),
        'SSP3': list(range(132, 144)),
        'SSP5': list(range(180, 192))
    },
    '2080-2099': {
        'Historical 1995-2014': list(range(0,12)),
        'SSP1': list(range(48, 60)),
        'SSP2': list(range(96, 108)),
        'SSP3': list(range(144, 156)),
        'SSP5': list(range(192, 204))
    }
}
# Define SSP colors
ssp_colors = {'SSP1': 'blue', 'SSP2': 'green', 'SSP3': 'orange', 'SSP5': 'red'}

# Define month labels
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']

# Define the figure and axes
fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4)

# Manually define the ranges for historical data
historical_range = list(range(0, 12))  # Adjust as per your actual data indices

# Plot Historical Data (Top Left)
axs[0, 0].plot(months, data.loc[historical_range, 'beta'], label="Historical 1995-2014", color="black")
axs[0, 0].set_title('Historical 1995-2014')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('Beta')
axs[0, 0].legend()

# Plot SSP Data (Bottom Rows)
for i, (period, ssp_dict) in enumerate(indices.items()):
    if i > 0:  # Skip the first item (historical data)
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2
        ax = axs[row, col]

        for ssp, idx_range in ssp_dict.items():
            period_data = data.iloc[idx_range].copy()
            period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
            period_data = period_data.sort_values('month')
            ax.plot(months, period_data['beta'], label=ssp, color=ssp_colors.get(ssp, 'black'))

        ax.set_title(f'Time Period: {period}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Beta')
        ax.legend()

        # Set x-ticks and labels for all subplots in the loop
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Assuming the dataframes and indices are already defined

input_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\Precipitation_temperature_Minas_Gerais_totAET.csv'
data = pd.read_csv(input_file_path)

indices = {
    '1995-2014': {
        'Historical 1995-2014': list(range(0,12)),
    },
    '2020-2039': {
        'Historical 1995-2014': list(range(0,12)),
        'SSP1': list(range(12, 24)),
        'SSP2': list(range(60, 72)),
        'SSP3': list(range(108, 120)),
        'SSP5': list(range(156, 168))
    },
    '2040-2059': {
        'Historical 1995-2014': list(range(0,12)),
        'SSP1': list(range(24, 36)),
        'SSP2': list(range(72, 84)),
        'SSP3': list(range(120, 132)),
        'SSP5': list(range(168, 180))
    },
    '2060-2079': {
        'Historical 1995-2014': list(range(0,12)),
        'SSP1': list(range(36, 48)),
        'SSP2': list(range(84, 96)),
        'SSP3': list(range(132, 144)),
        'SSP5': list(range(180, 192))
    },
    '2080-2099': {
        'Historical 1995-2014': list(range(0,12)),
        'SSP1': list(range(48, 60)),
        'SSP2': list(range(96, 108)),
        'SSP3': list(range(144, 156)),
        'SSP5': list(range(192, 204))
    }
}

# Define SSP colors
ssp_colors = {'SSP1': 'blue', 'SSP2': 'green', 'SSP3': 'orange', 'SSP5': 'red'}

# Define month labels
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']

# Define the figure and axes
fig, axs = plt.subplots(3, 2, figsize=(17, 18), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4)

# Manually define the ranges for historical data
historical_range = list(range(0, 12))  # Adjust as per your actual data indices

# Plot Historical Data (Top Left)
axs[0, 0].plot(months, data.loc[historical_range, 'AET'], label="AET with max_AET", color="blue")
axs[0, 0].plot(months, data.loc[historical_range, 'AET_without_max'], label="AET without max_AET", color="orange")
axs[0, 0].set_title('Historical 1995-2014')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('AET (mm/month)')
axs[0, 0].legend()

# Plot SSP Data (Bottom Rows)
for i, (period, ssp_dict) in enumerate(indices.items()):
    if i > 0:  # Skip the first item (historical data)
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2
        ax = axs[row, col]

        for ssp, idx_range in ssp_dict.items():
            period_data = data.iloc[idx_range].copy()
            period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
            period_data = period_data.sort_values('month')
            ax.plot(months, period_data['AET'], label=f"{ssp} AET with max_AET", color=ssp_colors.get(ssp, 'black'))
            ax.plot(months, period_data['AET_without_max'], label=f"{ssp} AET without max_AET", linestyle='--', color=ssp_colors.get(ssp, 'black'))

        ax.set_title(f'Time Period: {period}')
        ax.set_xlabel('Month')
        ax.set_ylabel('AET (mm/month)')
        ax.legend()

        # Set x-ticks and labels for all subplots in the loop
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)

plt.tight_layout()
plt.show()

