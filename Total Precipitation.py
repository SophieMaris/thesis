import pandas as pd 

# CALCULATE TOTAL PRECIPITATION PER STATE [m^3/month]

# File paths and area for each state
base_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python'
files = [
    {
        "input": f"{base_path}\Precipitation_temperature_Minas_Gerais.csv",
        "output": f"{base_path}\\total_precipitation\Precipitation_temperature_Minas_Gerais_totPrecipitation.csv",
        "area": 269481764694.99707 # Area Minas Gerais in m^2
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Bahia.csv",
        "output": f"{base_path}\\total_precipitation\Precipitation_temperature_Bahia_totPrecipitation.csv",
        "area": 298199130706.521 # Area Bahia in m^2
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Pernambuco.csv",
        "output": f"{base_path}\\total_precipitation\Precipitation_temperature_Pernambuco_totPrecipitation.csv",
        "area": 61848813793.99841  # Area Pernambuco in m^2
    }
]

# Process each file
for file_info in files:
    # Read precipitation data [mm/month] from the file
    data = pd.read_csv(file_info["input"])

    # Calculate total precipitation in [m^3/month] 
    data['total_precipitation'] = (data['precipitation'] / 1000) * file_info["area"]

    # Save the processed data to a new CSV file
    data.to_csv(file_info["output"], index=False)

    print(f"Processed file saved to: {file_info['output']}")



# CALCULATE TOTAL PRECIPITATION FOR ENTIRE BASIN [m^3/month]

# File paths of the processed files
file_paths = [
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\Precipitation_temperature_Minas_Gerais_totPrecipitation.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\Precipitation_temperature_Bahia_totPrecipitation.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\Precipitation_temperature_Pernambuco_totPrecipitation.csv'
]

# Read the data from each file into a list of dataframes
dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Extract the required columns from the first dataframe (assuming all have the same year, month, ssp columns)
result_df = dataframes[0][['year', 'month', 'ssp']].copy()

# Initialize the total_precipiation column in the result dataframe to zero
result_df['total_precipitation'] = 0

# Sum the total_precipitation columns across all dataframes
for df in dataframes:
    result_df['total_precipitation'] += df['total_precipitation']

# Save the result to a new CSV file
output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\total_precipitation_all_states.csv'
result_df.to_csv(output_file_path, index=False)

print(f"Summed total_precipitation saved to: {output_file_path}")


