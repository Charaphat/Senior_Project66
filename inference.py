import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, dropout=0.4, batch_first=True)
    self.fc = nn.Linear(hidden_size, 1)
  def forward(self, x):
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out

dataset = pd.read_csv('inference.csv')
X_test = dataset.iloc[:,:6].to_numpy()
y_test = dataset.iloc[:,6].to_numpy()

X_test = X_test.reshape((-1, X_test.shape[1], 1))
y_test = y_test.reshape((-1, 1))

X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

loaded_model = torch.load('models/epoch1800.pt')
model = LSTM(1,100,2)
model.load_state_dict(loaded_model['state_dict'])
model.to(device)
model.eval()

with torch.no_grad():
  predicted = model(X_test.to(device)).to('cpu').numpy()

# Save Figure
output_dir = 'LSTM'

os.makedirs(output_dir, exist_ok=True)

for i in range(0, len(predicted), 200):
  plt.figure(figsize=(12, 9))
  plt.clf()
  #plt.scatter(range(len(y_test[i:i+200,:])),y_test[i:i+200,:],label='Actual', color = 'blue', s=4)
  #plt.scatter(range(len(predicted[i:i+200,:])),predicted[i:i+200,:],label="Predicted", color = 'red', s=4)
  plt.plot(y_test[i:i+200,:],label='Actual', color='blue')
  plt.plot(predicted[i:i+200,:],label="Predicted", color='orange')
  plt.plot(abs(y_test[i:i+200,:] - predicted[i:i+200,:]), label="MAE loss", color='green')

  plt.xlabel("Hours")
  plt.ylabel("Temperature")
  plt.legend()

  filename = os.path.join(output_dir, f'hour{i}.png')
  plt.savefig(filename)

plt.close('all')
mae = mean_absolute_error(y_test,predicted)
print(mae)

