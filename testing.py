from util.dataset import *
import matplotlib.pyplot as plt
import pandas as pd

DIR = "./data/dataRaw.csv"
data = DatasetInterface()
data.initialize_dataset(DIR)

# Preprocessing
data.handle_outlier(type = 1, lenWin = 30) # Handle outliers (0 for IQR, 1 and lenWin > 1 for Isolation Forest)
data.data_normalization(lenDiff = 1) # Normalize data using Pandas difference method

# Create timeseries data with a split ratio of 80% training and 20% validation
data.create_timeseries(split = 0.8)

# plt.plot(data.df['price'])
fig, axs = plt.subplots(2, 2, figsize=(10, 4))  # 2x2 grid of subplots

# Plotting each DataFrame in a separate subplot
axs[0, 0].plot(data.target.pd_dataframe())
axs[0, 0].set_title('Target')

axs[0, 1].plot(data.targetScaled.pd_dataframe())
axs[0, 1].set_title('Train Target')

axs[1, 0].plot(data.valTarget.pd_dataframe())
axs[1, 0].set_title('Validation Target')

axs[1, 1].plot(data.df)
axs[1, 1].set_title('DF')

plt.tight_layout()
plt.show()