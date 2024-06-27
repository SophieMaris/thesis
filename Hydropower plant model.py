import pandas as pd
import matplotlib.pyplot as plt

# Constants
rho_w = 1000 
g = 9.81  
eta = 0.985  
h = 112.5  
f = 0.0075 
L = 94 
D = 8.7 
v = 6.4763803 

input_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\water_balance_model\total_discharge.csv'
data = pd.read_csv(input_file_path)

hydraulic_head_list = []
flow_velocity_list = []
hydropower_output_list = []

for index, row in data.iterrows():
    Q_t = row['total_discharge'] 

    h_f = f * (L / D) * (v**2) / (2 * g)  

    H = h - h_f
    hydraulic_head_list.append(H)

    P_t = Q_t * H * rho_w * g * eta / 1e6  
    hydropower_output_list.append(P_t)

data['hydraulic_head'] = hydraulic_head_list
data['hydropower_output'] = hydropower_output_list

output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\hydropower_plant_model\hydropower_plant_output.csv'
data.to_csv(output_file_path, index=False)

print(f"Hydropower output data saved to {output_file_path}")

csv_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\hydropower_plant_model\hydropower_plant_output.csv'  
data = pd.read_csv(csv_file_path)

excel_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\hydropower_plant_model\hydropower_plant_output.xlsx'  # Define the output Excel file path
data.to_excel(excel_file_path, index=False)

print(f"CSV file has been converted to Excel and saved as {excel_file_path}")

data = pd.read_csv(output_file_path)

# FIGURE 11: Potential power output historical

output_file_path = r'C:\Users\HP\Documents\Year 3\Thesis\Python\hydropower_plant_model\hydropower_plant_output.csv'
hydropower_output = pd.read_csv(output_file_path)

historical = hydropower_output.iloc[0:12]

plt.figure(figsize=(8, 5))

plt.plot(range(1, len(historical) + 1), historical["hydropower_output"], label="Historical", marker='o', color='black')

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plt.xticks(range(1, len(months) + 1), months, rotation=45)  # Set ticks and labels for each month

plt.xlabel("Month")
plt.ylabel("Potential Power Output (MW)")
plt.title("Potential Power Output 1995-2014")

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

# FIGURE 18: Potential power output with lines per time period

# Define the figure and axes
fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4)

# Define subplot labels
subplot_labels = ['a', 'b', 'c', 'd']

# Plot SSP Data (Bottom Rows)
for i, (period, ssp_dict) in enumerate(indices.items()):
    row = i // 2  # Calculate row index based on loop index
    col = i % 2   # Calculate column index based on loop index
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

    # Add subplot labels
    ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()

# FIGURE 13: Potential power output markers only per time period

# Define the figure and axes
fig, axs = plt.subplots(1, 4, figsize=(14.5, 6.5), sharex=True, sharey=True)  # 1 row, 4 columns
fig.subplots_adjust(wspace=0.3)  

# Define subplot labels
subplot_labels = ['a', 'b', 'c', 'd']

# Plot SSP Data (Four Columns)
for i, (period, ssp_dict) in enumerate(indices.items()):
    ax = axs[i]  # Use index i to select the correct subplot

    for ssp, idx_range in ssp_dict.items():
        period_data = data.iloc[idx_range].copy()
        period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
        period_data = period_data.sort_values('month')
        ax.plot(months, period_data['hydropower_output'], label=ssp, marker='o', linestyle='None', color=ssp_colors.get(ssp, 'black'), markersize=3)

    ax.set_title(f'Time Period: {period}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Potential Power Output (MW)')
    ax.legend()

    # Set x-ticks and labels for all subplots in the loop
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45)
    
    # Add subplot labels
    ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()

# FIGURE 20: Potential power output with lines subplot per SSP

# Define period colors with similar hues but different darkness levels
period_colors = {
    '2020-2039': '#9ecae1',   # Light blue
    '2040-2059': '#3182bd',   # Medium blue
    '2060-2079': '#08519c',   # Dark blue
    '2080-2099': '#08306b'    # Very dark blue
}

# Define the figure and axes
fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4)

# Loop through each SSP and plot its data in a separate subplot
for i, ssp in enumerate(['SSP1', 'SSP2', 'SSP3', 'SSP5']):
    row = i // 2  # Calculate row index based on loop index
    col = i % 2   # Calculate column index based on loop index
    ax = axs[row, col]

    # Plot historical data first
    historical_data = data.iloc[indices['2020-2039']['Historical 1995-2014']].copy()
    historical_data['month'] = pd.Categorical(historical_data['month'], categories=months, ordered=True)
    historical_data = historical_data.sort_values('month')
    ax.plot(months, historical_data['hydropower_output'], label='Historical 1995-2014', color='gray', linestyle='--')

    # Plot data for each period
    for period, ssp_dict in indices.items():
        if period != 'Historical 1995-2014':  # Skip historical data in this loop
            period_data = data.iloc[ssp_dict[ssp]].copy()
            period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
            period_data = period_data.sort_values('month')
            ax.plot(months, period_data['hydropower_output'], label=period, color=period_colors[period])

    ax.set_title(f'Subplot for SSP: {ssp}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Potential Power Output (MW)')
    ax.legend()

    # Set x-ticks and labels for all subplots in the loop
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45)

    # Add subplot label
    ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()

# FIGURE 15: Potential power output markers only subplots per SSP

# Define the figure and axes
fig, axs = plt.subplots(1, 4, figsize=(14.5, 6.5), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.3) 

# Loop through each SSP and plot its data in a separate subplot
for i, ssp in enumerate(['SSP1', 'SSP2', 'SSP3', 'SSP5']):
    ax = axs[i]

    # Plot historical data first
    historical_data = data.iloc[indices['2020-2039']['Historical 1995-2014']].copy()
    historical_data['month'] = pd.Categorical(historical_data['month'], categories=months, ordered=True)
    historical_data = historical_data.sort_values('month')
    ax.plot(months, historical_data['hydropower_output'], label='Historical 1995-2014', marker='o', color='gray', linestyle='None', markersize=3.5)

    # Plot data for each period
    for period, ssp_dict in indices.items():
        if period != 'Historical 1995-2014':  
            period_data = data.iloc[ssp_dict[ssp]].copy()
            period_data['month'] = pd.Categorical(period_data['month'], categories=months, ordered=True)
            period_data = period_data.sort_values('month')
            ax.plot(months, period_data['hydropower_output'], label=period, marker='o', linestyle='None', color=period_colors[period], markersize=3.5)

    ax.set_title(f'Subplot for SSP: {ssp}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Potential Power Output (MW)')
    ax.legend()

    # Set x-ticks and labels for all subplots in the loop
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45)

    # Add subplot label
    ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()