import segysak
import xarray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from segysak.segy import segy_loader, well_known_byte_locs, segy_writer
import segyio
data3d = xarray.open_dataset('data3d.SEISNC')
data2d = xarray.open_dataset('l-15.SEISNC')
# print(test.attrs['percentiles'])

data_values3d = data3d['data'].values
data_values2d = data2d['data'].values

# Assuming data is your 3D array
data3d = np.array(data3d['data'].values)
flattened_data3d = np.array(data3d).flatten()

data2d_ = np.array(data2d['data'].values)
flattened_data2d = np.array(data2d_).flatten()

mean_3d = sum(flattened_data3d) / len(flattened_data3d)
print(mean_3d)

# Menghitung deviasi standar dari data
std_dev3d = (sum((x-mean_3d) ** 2 for x in flattened_data3d) / len(flattened_data3d)) ** 0.5
print(std_dev3d)

# Count the frequency of each element
# frequency_counter = Counter(flattened_data3d)
# Convert Counter to dictionary
# frequency_dict = dict(frequency_counter)
# Convert dictionary to DataFrame
# df = pd.DataFrame(list(frequency_dict.items()), columns=['Element', 'Frequency'])

# scaler = MinMaxScaler(feature_range=(min(flattened_data3d), max(flattened_data3d)))
# flattened_data2d = scaler.fit_transform(flattened_data2d.reshape(-1, 1))

def standardize(data):
  # Menghitung rata-rata dari data
  mean_value = sum(data) / len(data)
  print(mean_value)

  # Menghitung deviasi standar dari data
  std_dev = (sum((x-mean_value) ** 2 for x in data) / len(data)) ** 0.5
  print(std_dev)

  # Menstandarisasi data menggunakan rumus (x - mean) / std_dev
  standardize_data = [(x - mean_value) / std_dev for x in data]

  # Mengembalikan data yang telah distandarisasi
  return standardize_data

standardize_data = standardize(flattened_data2d)

# scaling
flattened_data2d = [(x * std_dev3d) + mean_3d for x in standardize_data]

# print(flattened_data2d)
data2d['data'][:] = [[flattened_data2d]]
# print(data2d.attrs['text'])
# print(data2d.attrs['text'])
# segy_writer(data2d, "test2.segy")

# Assuming flattened_data2d is your data
fig, ax = plt.subplots()
n, bins, patches = ax.hist(flattened_data3d, bins=100, color='blue', alpha=0.5, density=True)

# Convert y-axis ticks to percentage
percentage_ticks = [tick * 100 for tick in ax.get_yticks()]
ax.set_yticklabels(['{:,.2%}'.format(tick) for tick in percentage_ticks])

n, bins, patches = ax.hist(flattened_data2d, bins=100, color='red', alpha=0.5, density=True)

# Convert y-axis ticks to percentage
percentage_ticks = [tick * 100 for tick in ax.get_yticks()]
ax.set_yticklabels(['{:,.2%}'.format(tick) for tick in percentage_ticks])

# ax.hist(flattened_data3d, bins=100, color='red', alpha=0.5, density=True)

plt.xlabel('Amplitude Value')
plt.ylabel('Percentage of Frequency')
plt.title('Histogram of Seismic Data (Percentage)')
plt.grid(True)
plt.show()