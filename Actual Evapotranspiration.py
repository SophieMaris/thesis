import math
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

month_to_num = {
    " January": 1, " February": 2, " March": 3, " April": 4, " May": 5,
    " June": 6, " July": 7, " August": 8, " September": 9, " October": 10,
    " November": 11, " December": 12
}

def calculate_ra(month, latitude_rad):

    month_num = month_to_num[month]
    
    day_of_year = 30.42 * month_num - 15.23
    
    d_r = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
    
    delta = 0.4093 * math.sin((2 * math.pi * day_of_year / 365) - 1.405)
    
    omega_s = math.acos(-math.tan(latitude_rad) * math.tan(delta))
 
    Ra = 15.392 * d_r * (omega_s * math.sin(latitude_rad) * math.sin(delta) +
                         math.cos(latitude_rad) * math.cos(delta) * math.sin(omega_s))
    return day_of_year, d_r, delta, omega_s, Ra

def calculate_pet(T_min, T_max, Ra, month_to_days):
    T_mean = (T_max + T_min) / 2
    PET = 0.0023 * Ra * (T_mean + 17.8) * math.sqrt(T_max - T_min) * month_to_days
    return PET

def calculate_max_aet(T_min, T_max, precipitation, coeff_avg_temp, coeff_precipitation):
    T_mean = (T_max + T_min) / 2
    max_AET = coeff_avg_temp * T_mean + coeff_precipitation * precipitation
    return max_AET

def perform_multiple_regression(file_path):

    data = pd.read_csv(file_path)
   
    y = data['avg_ssm']  
    X = data[['avg_temp', 'precipitation']] 

    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("______________________________________________________________________")
    print(vif_data)

    model = sm.OLS(y, X, hasconst=False).fit()

    print("______________________________________________________________________")
    print(model.summary())

    coeff_avg_temp = round(model.params['avg_temp'], 9)
    coeff_precipitation = round(model.params['precipitation'], 9)
    formula = f"avg_ssm = {coeff_avg_temp} * avg_temp + {coeff_precipitation} * precipitation"
    print("______________________________________________________________________")
    print("Linear Regression Model Formula:")
    print(formula)
    print("______________________________________________________________________")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Precipitation [mm/month]')
    ax.set_ylabel('Temperature [degrees Celcius]')
    ax.set_zlabel('Soil Moisture [mm/month]')
    plt.title('3D Scatter Plot of Soil Moisture, Temperature, and Precipitation')
    ax.scatter(data['precipitation'], data['avg_temp'], data['avg_ssm'], c='r', marker='o')
    plt.show()
   
    y_pred = model.predict(X)
    plt.scatter(y, y_pred)
    plt.xlabel('Actual Soil Moisture [mm/month]')
    plt.ylabel('Predicted Soil Moisture [mm/month]')
    plt.title('Actual vs Predicted Soil Moisture')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.show()
  
    residuals = y - y_pred
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Soil Moisture [mm/month]')
    plt.ylabel('Residuals [!!!!]')
    plt.title('Residuals vs Predicted Soil Moisture')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

    return coeff_avg_temp, coeff_precipitation

def calculate_total_aet(AET, area):
    return area * (AET / 1000) 

def calculate_beta(St_prev, St):
    ratio = St_prev / St
    beta = (5 * ratio - 2 * ratio**2) / 3
    return beta, ratio

def process_state_data(input_file, output_file, latitude, area):

    latitude_rad = latitude * math.pi / 180

    df = pd.read_csv(input_file)

    month_to_days = {
        ' January': 31,
        ' February': 28, 
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

    results = df.apply(lambda row: calculate_ra(row['month'], latitude_rad), axis=1)
    df[['day_of_month', 'd_r', 'delta', 'omega_s', 'Ra']] = pd.DataFrame(results.tolist(), index=df.index)
    df['PET'] = df.apply(lambda row: calculate_pet(row['Tmin'], row['Tmax'], row['Ra'], month_to_days[row['month']]), axis=1)
   
    file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\ssm_temperature_precipitation_historical_Brazil.csv'
    coeff_avg_temp, coeff_precipitation = perform_multiple_regression(file_path)

    df['max_AET'] = df.apply(lambda row: calculate_max_aet(row['Tmin'], row['Tmax'], row['precipitation'], coeff_avg_temp, coeff_precipitation), axis=1)

    St_prev = df['max_AET'].iloc[0]

    AET_list = []
    beta_list = []
    total_AET_list = []
    ratio_list = []
    AET_without_max_list = []
    for i, row in df.iterrows():
        St = row['max_AET']
        beta, ratio = calculate_beta(St_prev, St)

        AET_without_max = row['PET'] * beta

        AET = AET_without_max if AET_without_max <= row['max_AET'] else row['max_AET']
        
        total_AET = calculate_total_aet(AET, area)
        
        AET_list.append(AET)
        beta_list.append(beta)
        total_AET_list.append(total_AET)
        ratio_list.append(ratio)
        AET_without_max_list.append(AET_without_max)

        St_prev = St
    
base_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python'
files = [
    {
        "input": f"{base_path}\Precipitation_temperature_Minas_Gerais.csv",
        "output": f"{base_path}\\total_AET\Precipitation_temperature_Minas_Gerais_totAET.csv",
        "latitude": -17.8154808447, 
        "area": 269481764694.99707 
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Bahia.csv",
        "output": f"{base_path}\\total_AET\Precipitation_temperature_Bahia_totAET.csv",
        "latitude": -11.7474248842, 
        "area": 298199130706.521 
    },
    {
        "input": f"{base_path}\Precipitation_temperature_Pernambuco.csv",
        "output": f"{base_path}\\total_AET\Precipitation_temperature_Pernambuco_totAET.csv",
        "latitude": -8.2945583277, 
        "area": 61848813793.99841 
    }
]

for file in files:
    process_state_data(file['input'], file['output'], file['latitude'], file['area'])

file_paths = [
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\Precipitation_temperature_Minas_Gerais_totAET.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\Precipitation_temperature_Bahia_totAET.csv',
    r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\Precipitation_temperature_Pernambuco_totAET.csv'
]

dataframes = [pd.read_csv(file_path) for file_path in file_paths]

result_df = dataframes[0][['year', 'month', 'ssp']].copy()

result_df['total_AET'] = 0

for df in dataframes:
    result_df['total_AET'] += df['total_AET']

output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\total_AET_all_states.csv'
result_df.to_csv(output_file_path, index=False)

print(f"Summed total_AET saved to: {output_file_path}")


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

ssp_colors = {'SSP1': 'blue', 'SSP2': 'green', 'SSP3': 'orange', 'SSP5': 'red'}

months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']


# FIGURE 22: AET with and without capping at surface soil moisture availability

fig, axs = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4)

for i, (period, ssp_dict) in enumerate(indices.items()):
    if i > 0:
        row = (i - 1) // 2
        col = (i - 1) % 2
        ax = axs[row, col]

        for ssp, idx_range in ssp_dict.items():
            period_data = data.iloc[idx_range].copy()
            period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
            period_data = period_data.sort_values('month')
            ax.plot(months, period_data['beta'], label=ssp, linestyle='None', marker='o', color=ssp_colors.get(ssp, 'black'))

        ax.set_title(f'Time Period: {period}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Beta')
        ax.legend()

        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)

plt.tight_layout()
plt.show()

# FIGURE 23: Values for B for the state Minas Gerais, Brazil

fig, axs = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4)

for i, (period, ssp_dict) in enumerate(indices.items()):
    if i > 0:  
        row = (i - 1) // 2 
        col = (i - 1) % 2
        ax = axs[row, col]

        for ssp, idx_range in ssp_dict.items():
            period_data = data.iloc[idx_range].copy()
            period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
            period_data = period_data.sort_values('month')
            ax.plot(months, period_data['AET'], label=f"{ssp} AET with capping", linestyle='None', marker='o', color=ssp_colors.get(ssp, 'black'))
            ax.plot(months, period_data['AET_without_max'], label=f"{ssp} AET", linestyle='None', marker='*', color=ssp_colors.get(ssp, 'black'))

        ax.set_title(f'Time Period: {period}')
        ax.set_xlabel('Month')
        ax.set_ylabel('AET (mm/month)')
        ax.legend(fontsize='7')

        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)

plt.tight_layout()
plt.show()

