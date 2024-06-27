import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\susm_temperature_precipitation_historical_Brazil.csv'
data = pd.read_csv(file_path)

y = data['avg_susm']
X = data[['avg_temp', 'precipitation']] 

vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("______________________________________________________________________")
print(vif_data)

model = sm.OLS(y, X).fit()

print("______________________________________________________________________")
print(model.summary())

coeff_avg_temp = round(model.params['avg_temp'], 9)
coeff_precipitation = round(model.params['precipitation'], 9)
formula = f"avg_susm = {coeff_avg_temp} * avg_temp + {coeff_precipitation} * precipitation" 
print("______________________________________________________________________")
print("Linear Regression Model Formula (through origin):")
print(formula)
print("______________________________________________________________________")

y_pred = model.predict(X)

y_pred = y_pred.clip(lower=0)

# VISUALIZATION MLR

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Precipitation [mm/month]')
ax.set_ylabel('Temperature [degrees Celcius]')
ax.set_zlabel('Soil Moisture [mm/month]')
plt.title('3D Scatter Plot of Soil Moisture, Temperature, and Precipitation')
ax.scatter(data['precipitation'], data['avg_temp'], data['avg_susm'], c='r', marker='o')
plt.show()

plt.scatter(y, y_pred)
plt.xlabel('Actual Soil Moisture [mm/month]')
plt.ylabel('Predicted Soil Moisture [mm/month]')
plt.title('Actual vs Predicted Soil Moisture')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

residuals = y - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Soil Moisture [mm/month]')
plt.ylabel('Residuals [mm/month]')
plt.title('Residuals vs Predicted Soil Moisture')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

base_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python'
files = [
    {
        "input": f"{base_path}\Precipitation_temperature_Minas_Gerais.csv",
        "output": f"{base_path}\\total_change_in_storage\Precipitation_temperature_Minas_Gerais_totChangeInStorage.csv",
        "area": 269481764694.99707 
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Bahia.csv",
        "output": f"{base_path}\\total_change_in_storage\Precipitation_temperature_Bahia_totChangeInStorage.csv",
        "area": 298199130706.521 
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Pernambuco.csv",
        "output": f"{base_path}\\total_change_in_storage\Precipitation_temperature_Pernambuco_totChangeInStorage.csv",
        "area": 61848813793.99841 
    }
]

for file_info in files:

    data = pd.read_csv(file_info["input"])

    data['avg_temp'] = (data['Tmin'] + data['Tmax']) / 2 

    data['avg_susm'] =  coeff_avg_temp * data['avg_temp'] + coeff_precipitation * data['precipitation']

    data['total_susm'] = (data['avg_susm'] / 1000) * file_info["area"]

    data['total_change_in_storage'] = data['total_susm'].diff().fillna(0)

    data.to_csv(file_info["output"], index=False)

    print(f"Processed file saved to: {file_info['output']}")

file_paths = [
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\Precipitation_temperature_Minas_Gerais_totChangeInStorage.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\Precipitation_temperature_Bahia_totChangeInStorage.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\Precipitation_temperature_Pernambuco_totChangeInStorage.csv'
]

dataframes = [pd.read_csv(file_path) for file_path in file_paths]

result_df = dataframes[0][['year', 'month', 'ssp']].copy()

result_df['total_change_in_storage'] = 0

for df in dataframes:
    result_df['total_change_in_storage'] += df['total_change_in_storage']

output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\total_change_in_storage_all_states.csv'
result_df.to_csv(output_file_path, index=False)

print(f"Summed total_change_in_storage saved to: {output_file_path}")




