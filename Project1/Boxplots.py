#%%
import pandas as pd
import matplotlib.pyplot as plt

# Load in the data
# file = 'StandardizedDataFrameWithNansFilled.csv'
file_name = 'MachineLearning&DataMining/Project1/StandardizedDataFrameWithNansFilled.csv'
data = pd.read_csv(file_name)


# Create a figure with a grid layout
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create box plots for each numeric column
ax.boxplot(data.iloc[:, 4:].values, widths=0.2)

# Set custom x-axis ticks to match the columns you are plotting
xticks = range(len(data.columns[4:]))
ax.set_xticks(xticks)

# Adjust the layout to prevent overlapping titles
columns = [x.strip() for x in list(data.columns[4:])]
plt.xticks(xticks, columns, rotation=90)
plt.tight_layout()

# Show the combined figure with all box plots
plt.show()

# %%
