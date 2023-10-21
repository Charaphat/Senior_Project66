import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class DataProcessing:
  def __init__(self, path, data):
    self.path = path
    self.data = data
    self.read_csv()

  def read_csv(self):
    self.data = pd.read_csv(self.path)

def scatter_plot(x1, y1, label1, color1, size1, x2, y2, label2, color2, size2, xlabel, ylabel):
  plt.scatter(x1, y1, label=label1, color=color1, s=size1)
  plt.scatter(x2, y2, label=label2, color=color2, s=size2)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()

def line_plot(y, label, color, y2, label2, color2, xlabel, ylabel):
  plt.plot(y1, label=label1, color=color1)
  plt.plot(y2, label=label2, color=color2)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()



"""
air_temperature = DataProcessing("dataset/Air_temperature.csv", 'air_temperature')
global_solar_radiation = DataProcessing("dataset/Global_solar_radiation.csv", "global_solar_radiation")
precipitation = DataProcessing("dataset/Precipitation.csv", "precipitation")
wind_direction = DataProcessing("dataset/Wind_direction.csv", "precipitation")
wind_speed = DataProcessing("dataset/Wind_speed.csv", "wind_speed")
load_condition = DataProcessing("dataset/Load_condition.csv", "load_condition")
transformer_temperature = DataProcessing("dataset/Transformer_temperature.csv", "transformer_temperature")
"""

dataset = DataProcessing("dataset/full_data.csv", "dataset")
air_temperature = dataset.data.iloc[:,0]
global_solar_radiation = dataset.data.iloc[:,1]
precipitation = dataset.data.iloc[:,2]
wind_direction = dataset.data.iloc[:,4]
wind_speed = dataset.data.iloc[:,3]
load_condition = dataset.data.iloc[:,5]
transformer_temperature = dataset.data.iloc[:,6]

fig = plt.figure(figsize=(5, 3))

# Create a gridspec with 3 rows and 3 columns
gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 2])

# Define individual subplots
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[0, 2])
ax4 = plt.subplot(gs[1, 0])
ax5 = plt.subplot(gs[1, 1])
ax6 = plt.subplot(gs[1, 2])
ax7 = plt.subplot(gs[2, 0])
ax8 = plt.subplot(gs[2, 1])
ax9 = plt.subplot(gs[2, 2])

# Scatter plots and set titles for the visible subplots
ax1.scatter(range(len(air_temperature)), air_temperature)
ax1.set_title('Air Temperature')

ax2.scatter(range(len(global_solar_radiation)), global_solar_radiation)
ax2.set_title('Global Solar Radiation')

ax3.scatter(range(len(precipitation)), precipitation)
ax3.set_title('Precipitation')

ax4.scatter(range(len(wind_direction)), wind_direction)
ax4.set_title('Wind Direction')

ax5.scatter(range(len(wind_speed)), wind_speed)
ax5.set_title('Wind Speed')

ax6.scatter(range(len(load_condition)), load_condition)
ax6.set_title('Load Condition')

ax8.scatter(range(len(transformer_temperature)), transformer_temperature)
ax8.set_title('Transformer Temperature')

# Hide the subplots at ax1[2, 0] and ax1[2, 2]
ax7.axis('off')
ax9.axis('off')

# Adjust subplot spacing
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)

plt.show()

