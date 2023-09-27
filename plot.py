import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()


data = pd.read_csv('dataset/full_data.csv')
load_condition = pd.read_csv('dataset/Load_condition.csv')
transformer_temperature = pd.read_csv('dataset/Transformer_temperature.csv')


fig1, ax1 = plt.subplots(2,2)
ax1[0,0].scatter(range(len(data)), data['Air_temperature'])
ax1[0,0].set_title('Air Temperature')
ax1[0,1].scatter(range(len(data)), data['Transformer_temperature'])
ax1[0,1].set_title('Transformer Temperature')
ax1[1,0].scatter(range(len(data)), data['Load_condition'])
ax1[1,0].set_title('Load Condition')
ax1[1,1].scatter(range(len(data)), data['Global_solar_radiation'])
ax1[1,1].set_title('Global Solar Radiation')


fig2, ax2 = plt.subplots(2,2)
ax2[0,0].scatter(range(len(load_condition)), load_condition['Load_condition'])
ax2[0,0].set_title('Load Condition')
ax2[0,1].scatter(range(len(transformer_temperature)), transformer_temperature['Transformer_temperature'])
ax2[0,1].set_title('Transformer Temperature')

plt.show()
