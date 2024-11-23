import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
ece3_data = pd.read_csv('/home/ubuntu/data/ML/results/OCEANIDS/bremerhaven-ece3-pred-table.csv')
era5_data = pd.read_csv('/home/ubuntu/data/ML/results/OCEANIDS/bremerhaven-era5-pred-table.csv')

# Merge the data on the prediction type
merged_data = pd.merge(ece3_data, era5_data, on='pred', suffixes=('_ece3', '_era5'))

# Create bar diagram for all prediction types
labels = merged_data['pred']
x = range(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 8))

# Plot RMSE values
ece3_rmse_values = merged_data['RMSE_ece3']
era5_rmse_values = merged_data['RMSE_era5']
rects1 = ax.bar(x, ece3_rmse_values, width, label='ece3 RMSE', color='darkturquoise')
rects2 = ax.bar([p + width for p in x], era5_rmse_values, width, label='era5 RMSE', color='darkcyan')

# Plot MAE values
ece3_mae_values = merged_data['MAE_ece3']
era5_mae_values = merged_data['MAE_era5']
rects3 = ax.bar([p + 2 * width for p in x], ece3_mae_values, width, label='ece3 MAE', color='limegreen')
rects4 = ax.bar([p + 3 * width for p in x], era5_mae_values, width, label='era5 MAE', color='forestgreen')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Prediction Types')
ax.set_ylabel('Values')
ax.set_title('Comparison of Predictions for Bremerhaven')
ax.set_xticks([p + 1.5 * width for p in x])
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

# Attach a text label above each bar in rects, displaying its height.
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
plt.show()
plt.savefig('/home/ubuntu/data/ML/results/OCEANIDS/bremerhaven-pred-comparison.png')