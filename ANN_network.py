import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import mean_squared_error
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

class DataProcessing:
  def __init__(self, path, data):
    self.path = path
    self.data = data
    self.read_csv()
    self.fill_missing_value()

  def read_csv(self):
    self.data = pd.read_csv(self.path)

  def fill_missing_value(self):
    for i in range(len(self.data)):
      if pd.isna(self.data.iloc[i,1]):
        self.data.iloc[i,1] = 0

  def data_info(self):
    print(f"\n#################### {self.data.columns[1]} ####################")
    print(self.data.info())
    for col in self.data.columns:
      null_lines = self.data.index[self.data[col].isnull().sum()]
      print(f"Sum of null : {null_lines}")

class ANN(nn.Module):
  def __init__(self):
    super().__init__()
    self.dropout = nn.Dropout(0)
    self.input_layer = nn.Linear(in_features=6, out_features=5)
    self.activation_function1 = nn.ReLU()
    self.hidden_layer1 = nn.Linear(in_features=5, out_features=2)
    self.activation_function2 = nn.ReLU()
    self.output_layer = nn.Linear(in_features=2, out_features=1)
    self.activation_function3 = nn.ReLU()

  def forward(self, x):
    x = self.dropout(x)
    x = self.activation_function1(self.input_layer(x))
    x = self.activation_function2(self.hidden_layer1(x))
    x = self.activation_function3(self.output_layer(x))
    return x

dataset = DataProcessing('dataset/full_data.csv', "data")

X = dataset.data.iloc[:,:6]
y = dataset.data.iloc[:,6]

split_index = int(len(X) * 0.85)

X_train = X[:split_index].to_numpy()
X_test = X[split_index:].to_numpy()
y_train = y[:split_index].to_numpy()
y_test = y[split_index:].to_numpy()

X_train = X_train.reshape((-1, X_train.shape[1]))
X_test = X_test.reshape((-1, X_test.shape[1]))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 370
learning_rate = 0.01

model = ANN().to(device)
loss_function = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
  model.train()
  y_pred = model(X_train).squeeze()
  loss = loss_function(y_pred, y_train)
  #print(loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  model.eval()
  with torch.no_grad():
    test_logits = model(X_test).squeeze()
    test_loss = loss_function(test_logits, y_test)

  if epoch % 1 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

with torch.no_grad():
  predicted = model(X_test.to(device)).to('cpu').numpy()


y_test = y_test.to('cpu').numpy()


output_dir = 'ANN'

os.makedirs(output_dir, exist_ok=True)

for i in range(0, len(predicted), 200):
  plt.clf()
  print(i)
  plt.plot(y_test[i:i+200,:],label='Actual')
  plt.plot(predicted[i:i+200,:],label="Predicted")
  plt.xlabel("Hours")
  plt.ylabel("Temperature")
  plt.legend()
  filename = os.path.join(output_dir, f'hour{i}.png')
  plt.savefig(filename)

# Clean up: Close all figures
plt.close('all')
mse = mean_squared_error(y_test, predicted)
print(mse)

