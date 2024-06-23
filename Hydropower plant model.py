import pandas as pd
import matplotlib.pyplot as plt

# Constants
rho_w = 1000  # Water’s density [kg/m^3]
g = 9.81  # gravitational acceleration [m/s^2]
eta = 0.985  # Turbine’s efficiency [-]
h = 112.5  # Hydraulic head [m]
f = 0.0075  # Friction factor [-]
L = 94  # Pipe length [m] 
D = 8.7  # Pipe’s internal diameter [m] 
v = 6.4763803 # Flow velocity in pipe [m/s]

# Read the input CSV file
input_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\water_balance_model\total_discharge.csv'
data = pd.read_csv(input_file_path)

# Initialize lists to store results
hydraulic_head_list = []
flow_velocity_list = []
hydropower_output_list = []

# Process each row
for index, row in data.iterrows():
    Q_t = row['total_discharge']  # Input flow [m^3/s]

    # Calculate head loss due to friction
    h_f = f * (L / D) * (v**2) / (2 * g)  # head loss [m]

    # Calculate the effective head of the turbine
    H = h - h_f
    hydraulic_head_list.append(H)

    # Calculate the hydropower output
    P_t = Q_t * H * rho_w * g * eta / 1e6  # Convert to MW
    hydropower_output_list.append(P_t)

# Add the calculated values to the DataFrame
data['hydraulic_head'] = hydraulic_head_list
data['hydropower_output'] = hydropower_output_list

# Save the results to a new CSV file
output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\hydropower_plant_model\hydropower_plant_output.csv'
data.to_csv(output_file_path, index=False)

print(f"Hydropower output data saved to {output_file_path}")

import pandas as pd

# Load the CSV file into a DataFrame
csv_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\hydropower_plant_model\hydropower_plant_output.csv'  
data = pd.read_csv(csv_file_path)

# Save the DataFrame to an Excel file
excel_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\hydropower_plant_model\hydropower_plant_output.xlsx'  # Define the output Excel file path
data.to_excel(excel_file_path, index=False)

print(f"CSV file has been converted to Excel and saved as {excel_file_path}")

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
axs[0, 0].plot(months, data.loc[historical_range, 'hydropower_output'], label="Historical 1995-2014", color="black")
axs[0, 0].set_title('Historical 1995-2014')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('Potential Power Output (MW)')
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
            ax.plot(months, period_data['hydropower_output'], label=ssp, color=ssp_colors.get(ssp, 'black'))

        ax.set_title(f'Time Period: {period}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Potential Power Output (MW)')
        ax.legend()

        # Set x-ticks and labels for all subplots in the loop
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)

plt.tight_layout()
plt.show()
