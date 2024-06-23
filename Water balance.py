import pandas as pd

# File paths for input files
precipitation_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\total_precipitation_all_states.csv'
AET_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\total_AET_all_states.csv'
change_in_storage_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\total_change_in_storage_all_states.csv'

# Read input files
precipitation_df = pd.read_csv(precipitation_file) 
AET_df = pd.read_csv(AET_file)
change_in_storage_df = pd.read_csv(change_in_storage_file)

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

# Add a column to map month names to days in each month
precipitation_df['days_in_month'] = precipitation_df['month'].map(month_to_days)

# Calculate total discharge [m^3/s]
total_discharge_df = pd.DataFrame({
    'year': precipitation_df['year'],
    'month': precipitation_df['month'],
    'ssp': precipitation_df['ssp'],
    'total_discharge': (precipitation_df['total_precipitation'] - AET_df['total_AET'] + change_in_storage_df['total_change_in_storage'])/ (precipitation_df['days_in_month']*24*60*60)
})

# Write to CSV
output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\water_balance_model\total_discharge.csv'
total_discharge_df.to_csv(output_file_path, index=False)

print(f"Total discharge file saved to: {output_file_path}")

import pandas as pd

# Load the CSV file into a DataFrame
csv_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\water_balance_model\total_discharge.csv'  # Replace with your actual CSV file path
data = pd.read_csv(csv_file_path)

# Save the DataFrame to an Excel file
excel_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\water_balance_model\total_discharge.xlsx'  # Define the output Excel file path
data.to_excel(excel_file_path, index=False)

print(f"CSV file has been converted to Excel and saved as {excel_file_path}")

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
total_discharge = pd.read_csv(output_file_path)

# Step 2: Extract Data for Each SSP and Time Period 2080-2099
historical = total_discharge.iloc[2:14]
ssp1 = total_discharge.iloc[50:62]
ssp2 = total_discharge.iloc[98:110]
ssp3 = total_discharge.iloc[146:158]
ssp5 = total_discharge.iloc[194:206]

# Step 3: Plotting
plt.figure(figsize=(12, 6))

# Plot Historical Data
plt.plot(range(1, len(historical) + 1), historical["total_discharge"], label="Historical")

# Plot SSP1 Data
plt.plot(range(1, len(ssp1) + 1), ssp1["total_discharge"], label="SSP1", color="blue")

# Plot SSP2 Data
plt.plot(range(1, len(ssp2) + 1), ssp2["total_discharge"], label="SSP2", color="green")

# Plot SSP3 Data
plt.plot(range(1, len(ssp3) + 1), ssp3["total_discharge"], label="SSP3", color="orange")

# Plot SSP5 Data
plt.plot(range(1, len(ssp5) + 1), ssp5["total_discharge"], label="SSP5", color="red")

# Customize the plot
plt.xlabel("Row Number")
plt.ylabel("Discharge")
plt.title("Total Discharge for Different SSPs")
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(output_file_path)

# Define row indices for each time period and SSP
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


import pandas as pd
import matplotlib.pyplot as plt

# Assuming data, indices, months, and ssp_colors are already defined

# Define SSP colors
ssp_colors = {'SSP1': 'blue', 'SSP2': 'green', 'SSP3': 'orange', 'SSP5': 'red'}

# Define month labels
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Define the figure and axes
fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4)

# Manually define the ranges for historical data
historical_range = list(range(0, 12))  # Adjust as per your actual data indices

# Plot Historical Data (Top Left)
axs[0, 0].plot(months, data.loc[historical_range, 'total_discharge'], label="Historical 1995-2014", color="black")
axs[0, 0].set_title('Historical 1995-2014')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('Total Discharge (m³/s)')
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
            ax.plot(months, period_data['total_discharge'], label=ssp, color=ssp_colors.get(ssp, 'black'))

        ax.set_title(f'Time Period: {period}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Total Discharge (m³/s)')
        ax.legend()

        # Set x-ticks and labels for all subplots in the loop
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# File paths for input files
precipitation_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_precipitation\total_precipitation_all_states.csv'
AET_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_AET\total_AET_all_states.csv'
change_in_storage_file = r'C:\Users\HP\Documents\Year 3\Thesis\Python\total_change_in_storage\total_change_in_storage_all_states.csv'

# Read input files
precipitation_df = pd.read_csv(precipitation_file)
AET_df = pd.read_csv(AET_file)
change_in_storage_df = pd.read_csv(change_in_storage_file)

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

# Add a column to map month names to days in each month
precipitation_df['days_in_month'] = precipitation_df['month'].map(month_to_days)

# Define time period index ranges for 2080-2099
time_period_indices = list(range(192, 204))  # Adjust according to your data

# Extract data for 2080-2099
precipitation_data = precipitation_df.iloc[time_period_indices]
AET_data = AET_df.iloc[time_period_indices]
change_in_storage_data = change_in_storage_df.iloc[time_period_indices]

# Convert monthly values to daily values
precipitation_data['total_precipitation'] /= precipitation_data['days_in_month']
AET_data['total_AET'] /= precipitation_data['days_in_month']
change_in_storage_data['total_change_in_storage'] /= precipitation_data['days_in_month']

# Convert daily values to seconds
precipitation_data['total_precipitation'] /= 24 * 60 * 60
AET_data['total_AET'] /= 24 * 60 * 60
change_in_storage_data['total_change_in_storage'] /= 24 * 60 * 60

# Define months for x-axis
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Create a figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# Plotting precipitation
ax1.plot(months, precipitation_data['total_precipitation'], marker='o', linestyle='-', color='b')
ax1.set_title('Total Precipitation (2080-2099)')
ax1.set_ylabel('Precipitation (m³/s)')
ax1.grid(True)

# Plotting AET
ax2.plot(months, AET_data['total_AET'], marker='o', linestyle='-', color='g')
ax2.set_title('Actual Evapotranspiration (2080-2099)')
ax2.set_ylabel('AET (m³/s)')
ax2.grid(True)

# Plotting Change in Storage
ax3.plot(months, change_in_storage_data['total_change_in_storage'], marker='o', linestyle='-', color='r')
ax3.set_title('Change in Soil Moisture (2080-2099)')
ax3.set_xlabel('Month')
ax3.set_ylabel('Change in Soil Moisture (m³/s)')
ax3.grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()







