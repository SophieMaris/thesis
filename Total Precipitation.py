import pandas as pd 

base_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python'
files = [
    {
        "input": f"{base_path}\Precipitation_temperature_Minas_Gerais.csv",
        "output": f"{base_path}\\total_precipitation\Precipitation_temperature_Minas_Gerais_totPrecipitation.csv",
        "area": 269481764694.99707 
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Bahia.csv",
        "output": f"{base_path}\\total_precipitation\Precipitation_temperature_Bahia_totPrecipitation.csv",
        "area": 298199130706.521 
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Pernambuco.csv",
        "output": f"{base_path}\\total_precipitation\Precipitation_temperature_Pernambuco_totPrecipitation.csv",
        "area": 61848813793.99841
    }
]

for file_info in files:

    data = pd.read_csv(file_info["input"])

    data['total_precipitation'] = (data['precipitation'] / 1000) * file_info["area"]

    data.to_csv(file_info["output"], index=False)

    print(f"Processed file saved to: {file_info['output']}")

file_paths = [
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\Precipitation_temperature_Minas_Gerais_totPrecipitation.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\Precipitation_temperature_Bahia_totPrecipitation.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\Precipitation_temperature_Pernambuco_totPrecipitation.csv'
]

dataframes = [pd.read_csv(file_path) for file_path in file_paths]

result_df = dataframes[0][['year', 'month', 'ssp']].copy()

result_df['total_precipitation'] = 0

for df in dataframes:
    result_df['total_precipitation'] += df['total_precipitation']

output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\total_precipitation_all_states.csv'
result_df.to_csv(output_file_path, index=False)

print(f"Summed total_precipitation saved to: {output_file_path}")


