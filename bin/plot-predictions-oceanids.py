import pandas as pd
import matplotlib.pyplot as plt
import sys

file_path = sys.argv[1]

# Read the CSV file
df = pd.read_csv(file_path)

# Set the 'utctime' column as the index
df.set_index('utctime', inplace=True)

# Plot the data
df.plot(figsize=(10, 6))
plt.title('Predictions vs Training data')
plt.xlabel('Date')
plt.ylabel(file_path.replace('-prediction.csv', ''))
plt.legend(title='Columns')
plt.grid(True)
plt.show()
plt.savefig(file_path.replace('.csv', '.png'))