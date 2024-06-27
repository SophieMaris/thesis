import pandas as pd
import matplotlib.pyplot as plt

precipitation_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\total_precipitation_all_states.csv'
AET_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\total_AET_all_states.csv'
change_in_storage_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\total_change_in_storage_all_states.csv'

precipitation_df = pd.read_csv(precipitation_file) 
AET_df = pd.read_csv(AET_file)
change_in_storage_df = pd.read_csv(change_in_storage_file)

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

precipitation_df['days_in_month'] = precipitation_df['month'].map(month_to_days)

total_discharge_list = []
for idx, row in precipitation_df.iterrows():
    year = row['year']
    month = row['month']
    ssp = row['ssp']
    total_precipitation = row['total_precipitation']
    total_AET = AET_df.loc[idx, 'total_AET']
    total_change_in_storage = change_in_storage_df.loc[idx, 'total_change_in_storage']
    days_in_month = row['days_in_month']
    
    total_discharge_monthly= (total_precipitation - total_AET + total_change_in_storage)

    total_discharge = (total_precipitation - total_AET + total_change_in_storage) / (days_in_month * 24 * 60 * 60)
    
    total_discharge_list.append([year, month, ssp, days_in_month, total_discharge, total_discharge_monthly])

total_discharge_df = pd.DataFrame(total_discharge_list, columns=['year', 'month', 'ssp', 'days_in_month', 'total_discharge', 'total_discharge_monthly'])

# Write to CSV
output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\water_balance_model\total_discharge.csv'
total_discharge_df.to_csv(output_file_path, index=False)

print(f"Total discharge file saved to: {output_file_path}")


csv_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\water_balance_model\total_discharge.csv'  # Replace with your actual CSV file path
data = pd.read_csv(csv_file_path)


excel_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\water_balance_model\total_discharge.xlsx'  # Define the output Excel file path
data.to_excel(excel_file_path, index=False)

print(f"CSV file has been converted to Excel and saved as {excel_file_path}")

output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\water_balance_model\total_discharge.csv'
total_discharge = pd.read_csv(output_file_path)


# FIGURE 10: Historical discharge

historical = total_discharge.iloc[0:12]

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(historical) + 1), historical["total_discharge"], label="Historical", marker='o', color='black')

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plt.xticks(range(1, len(months) + 1), months, rotation=45)  # Set ticks and labels for each month
plt.xlabel("Month")
plt.ylabel("Discharge (m³/s)")
plt.title("Total Discharge for 1995-2014")

plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()


indices = {
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
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# FIGURE 17: Discharge with lines per time period

fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4)

subplot_labels = ['a', 'b', 'c', 'd']

for i, (period, ssp_dict) in enumerate(indices.items()):
    row = i // 2  
    col = i % 2   
    ax = axs[row, col]

    for ssp, idx_range in ssp_dict.items():
        period_data = data.iloc[idx_range].copy()
        period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
        period_data = period_data.sort_values('month')
        ax.plot(months, period_data['total_discharge'], label=ssp, color=ssp_colors.get(ssp, 'black'))

    ax.set_title(f'Time Period: {period}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Discharge (m³/s)')
    ax.legend()

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45)

    ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()

# FIGURE 12: Discharge markers only per time period

fig, axs = plt.subplots(1, 4, figsize=(14.5, 6.5), sharex=True, sharey=True)  # 1 row, 4 columns
fig.subplots_adjust(wspace=0.3)  

subplot_labels = ['a', 'b', 'c', 'd']

for i, (period, ssp_dict) in enumerate(indices.items()):
    ax = axs[i]  

    for ssp, idx_range in ssp_dict.items():
        period_data = data.iloc[idx_range].copy()
        period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
        period_data = period_data.sort_values('month')
        ax.plot(months, period_data['total_discharge'], label=ssp, marker='o', linestyle='None', color=ssp_colors.get(ssp, 'black'), markersize=3)

    ax.set_title(f'Time Period: {period}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Discharge (m³/s)')
    ax.legend()


    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45)
    
    ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()


# FIGURE 19: Discharge with lines per SSP

period_colors = {
    '2020-2039': '#9ecae1',   # Light blue
    '2040-2059': '#3182bd',   # Medium blue
    '2060-2079': '#08519c',   # Dark blue
    '2080-2099': '#08306b'    # Very dark blue
}

fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4)

for i, ssp in enumerate(['SSP1', 'SSP2', 'SSP3', 'SSP5']):
    row = i // 2  
    col = i % 2   
    ax = axs[row, col]

    historical_data = data.iloc[indices['2020-2039']['Historical 1995-2014']].copy()
    historical_data['month'] = pd.Categorical(historical_data['month'], categories=months, ordered=True)
    historical_data = historical_data.sort_values('month')
    ax.plot(months, historical_data['total_discharge'], label='Historical 1995-2014', color='gray', linestyle='--')

    for period, ssp_dict in indices.items():
        if period != 'Historical 1995-2014':  
            period_data = data.iloc[ssp_dict[ssp]].copy()
            period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
            period_data = period_data.sort_values('month')
            ax.plot(months, period_data['total_discharge'], label=period, color=period_colors[period])

    ax.set_title(f'Subplot for SSP: {ssp}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Discharge (m³/s)')
    ax.legend()

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45)

    ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()


# FIGURE 14: Discharge markers only per SSP

fig, axs = plt.subplots(1, 4, figsize=(14.5, 6.5), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.3) 

for i, ssp in enumerate(['SSP1', 'SSP2', 'SSP3', 'SSP5']):
    ax = axs[i]

    historical_data = data.iloc[indices['2020-2039']['Historical 1995-2014']].copy()
    historical_data['month'] = pd.Categorical(historical_data['month'], categories=months, ordered=True)
    historical_data = historical_data.sort_values('month')
    ax.plot(months, historical_data['total_discharge'], label='Historical 1995-2014', marker='o', color='gray', linestyle='None', markersize=3.5)

    for period, ssp_dict in indices.items():
        if period != 'Historical 1995-2014': 
            period_data = data.iloc[ssp_dict[ssp]].copy()
            period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
            period_data = period_data.sort_values('month')
            ax.plot(months, period_data['total_discharge'], label=period, marker='o', linestyle='None', color=period_colors[period], markersize=3.5)

    ax.set_title(f'Subplot for SSP: {ssp}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Discharge (m³/s)')
    ax.legend()

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45)

    ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()


# FIGURE 16: Components of the water balance

precipitation_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\total_precipitation_all_states.csv'
AET_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\total_AET_all_states.csv'
change_in_storage_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\total_change_in_storage_all_states.csv'

precipitation_df = pd.read_csv(precipitation_file)
AET_df = pd.read_csv(AET_file)
change_in_storage_df = pd.read_csv(change_in_storage_file)

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

precipitation_df['days_in_month'] = precipitation_df['month'].map(month_to_days)

time_period_indices = list(range(0, 12))

precipitation_data = precipitation_df.iloc[time_period_indices]
AET_data = AET_df.iloc[time_period_indices]
change_in_storage_data = change_in_storage_df.iloc[time_period_indices]

precipitation_data['total_precipitation'] /= precipitation_data['days_in_month']
AET_data['total_AET'] /= precipitation_data['days_in_month']
change_in_storage_data['total_change_in_storage'] /= precipitation_data['days_in_month']

precipitation_data['total_precipitation'] /= 24 * 60 * 60
AET_data['total_AET'] /= 24 * 60 * 60
change_in_storage_data['total_change_in_storage'] /= 24 * 60 * 60

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

ax1.plot(months, precipitation_data['total_precipitation'], marker='o', linestyle='-', color='b')
ax1.set_title('Total Precipitation')
ax1.set_ylabel('Precipitation (m³/s)')
ax1.grid(True)

ax2.plot(months, AET_data['total_AET'], marker='o', linestyle='-', color='g')
ax2.set_title('Total Actual Evapotranspiration')
ax2.set_ylabel('AET (m³/s)')
ax2.grid(True)

ax3.plot(months, change_in_storage_data['total_change_in_storage'], marker='o', linestyle='-', color='r')
ax3.set_title('Total Change in Soil Moisture')
ax3.set_xlabel('Month')
ax3.set_ylabel('Change in Soil Moisture (m³/s)')
ax3.grid(True)

plt.suptitle('Components of Water Balance Model: Historical data (1995-2014)', fontsize=20)
plt.tight_layout()
plt.show()







