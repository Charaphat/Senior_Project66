import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataProcessing:
  def __init__(self, path, data):
    self.path = path
    self.data = data
    self.null_lines = []
    self.read_csv()
    self.data_info()
    #self.fill_missing_value()

  def read_csv(self):
    self.data = pd.read_csv(self.path)

  def fill_missing_value(self):
    for i in range(len(self.data)):
      if pd.isna(self.data.iloc[i,1]):
        self.data.iloc[i,1] = 0

  def data_info(self):
    #print(f"\n#################### {self.data.columns[1]} ####################")
    #print(self.data.info())
    for col in self.data.columns:
      null_lines = self.data.index[self.data[col].isnull()]
      #print(f"Sum of null : {null_lines}")
    for i in range(len(null_lines)):
      self.null_lines.append(null_lines[i])

# Read data
air_temperature = DataProcessing("dataset/Air_temperature.csv", 'air_temperature')
global_solar_radiation = DataProcessing("dataset/Global_solar_radiation.csv", "global_solar_radiation")
precipitation = DataProcessing("dataset/Precipitation.csv", "precipitation")
wind_direction = DataProcessing("dataset/Wind_direction.csv", "precipitation")
wind_speed = DataProcessing("dataset/Wind_speed.csv", "wind_speed")
load_condition = DataProcessing("dataset/Load_condition.csv", "load_condition")
transformer_temperature = DataProcessing("dataset/Transformer_temperature.csv", "transformer_temperature")

# Preprocessing in Load condition filter value 0 and  50 < load_condition && load_condtion > 625 
load_condition_drop_index = []
for index, value in enumerate(load_condition.data.iloc[:,1]):
  if value == 0:
    load_condition_drop_index.append(index)
  if value > 625 or value < 50:
    load_condition_drop_index.append(index)

# Preprocessing in Global solar radiation
global_solar_radiation_drop_index = []
for index, value in enumerate(global_solar_radiation.data.iloc[:,1]):
    if value > 3600:
        global_solar_radiation_drop_index.append(index)

precipitation_drop_index = []
for index, value in enumerate(precipitation.data.iloc[:,1]):
    if value > 9:
        precipitation_drop_index.append(index)

# Preprocessing in Tranformer temperature filter condition more than 70 until it reach to less than 55 again
# Drop index 16800 to 17200 because Transformer was maintained
start_index = None
end_index = None
transformer_temperature_drop_index = []
for i, value in enumerate(transformer_temperature.data.iloc[:,1]):
    if value >= 57:
        if start_index is None:
            start_index = i
    elif value < 55 and start_index is not None:
        end_index = i
        for num in range(start_index, end_index + 1):
          transformer_temperature_drop_index.append(num)
        start_index = None
    if i > 16800 and i < 17200:
        if value < 30:
            transformer_temperature_drop_index.append(i)
    if value < 27:
        transformer_temperature_drop_index.append(i)

# Union all index that was prepared to drop in dataframe
all_null_lines_set = set(precipitation_drop_index) | set(global_solar_radiation_drop_index) | set(transformer_temperature_drop_index) | set(air_temperature.null_lines) | set(global_solar_radiation.null_lines) | set(precipitation.null_lines) | set(wind_direction.null_lines) | set(wind_speed.null_lines) | set(load_condition.null_lines) | set(load_condition_drop_index)
all_null_lines = list(all_null_lines_set)
all_null_lines.sort(reverse=True)

# Combine all data to the same dataframe
X = pd.DataFrame(air_temperature.data.iloc[:,1])
X["Global_solar_radiation"] = global_solar_radiation.data.iloc[:,1]
X["Load_condition"] = load_condition.data.iloc[:,1]
X["Precipitation"] = precipitation.data.iloc[:,1]
X["Wind_speed"] = wind_speed.data.iloc[:,1]
X["Wind_direction"] = wind_direction.data.iloc[:,1]
X["Transformer_temperature"] = transformer_temperature.data.iloc[:,1]

# check null and missing value in dataframe before filter
X_missing_value = X.isna().sum()

# dataframe after filter
X_drop_null = X.drop(all_null_lines)

# check null and missing value in dataframe after filter
X_drop_null_missing_value = X_drop_null.isna().sum()

# Save dataframe to a new csv file
X_drop_null.to_csv('dataset/full_data.csv', index=False)

