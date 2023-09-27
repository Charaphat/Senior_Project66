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
torch.manual_seed(42)

class DataProcessing:
  def __init__(self, path, data):
    self.path = path
    self.data = data
    self.read_csv()

  def read_csv(self):
    self.data = pd.read_csv(self.path)

  def data_info(self):
    print(f"\n#################### {self.data.columns[1]} ####################")
    print(self.data.info())
    for col in self.data.columns:
      null_lines = self.data.index[self.data[col].isnull().sum()]
      print(f"Sum of null : {null_lines}")

class TimeSeriesDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y
  def __len__(self):
    return len(self.X)
  def __getitem__(self, i):
    return self.X[i], self.y[i]

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

class biLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, dropout=0.4, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(hidden_size*2, 1)
  def forward(self, x):
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_stacked_layers*2, batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(self.num_stacked_layers*2, batch_size, self.hidden_size).to(device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(GRU, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.4
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, _ = self.gru(x, h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.4
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, h0 = self.rnn(x, h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_one_epoch():
  model.train(True)
  print(f'Epoch: {epoch + 1}')
  running_loss = 0.0
  for batch_index, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    output = model(x_batch)
    loss = loss_function(output, y_batch)
    running_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % 100 == 99:
      avg_loss_across_batches = running_loss / 100
      print('Batch {0}, Loss: {1:.2f}'.format(batch_index+1, avg_loss_across_batches))
      running_loss = 0.0
  print()

def validate_one_epoch():
  model.train(False)
  running_loss = 0.0
  for batch_index, batch in enumerate(test_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    with torch.no_grad():
      output = model(x_batch)
      loss = loss_function(output, y_batch)
      running_loss += loss.item()
  avg_loss_across_batches = running_loss / len(test_loader)
  print('Val Loss: {0:.2f}'.format(avg_loss_across_batches))
  print('***************************************************')
  print()

dataset = DataProcessing('dataset/full_data.csv', "data")

X = dataset.data.iloc[:,:6]
y = dataset.data.iloc[:,6]

split_index = int(len(X) * 0.8)

X_train = X[:split_index].to_numpy()
X_test = X[split_index:].to_numpy()
y_train = y[:split_index].to_numpy()
y_test = y[split_index:].to_numpy()

X_train = X_train.reshape((-1, X_train.shape[1], 1))
X_test = X_test.reshape((-1, X_test.shape[1], 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = biLSTM(1, 150, 3)
model.to(device)

learning_rate = 0.0001
num_epochs = 15
loss_function = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  train_one_epoch()
  validate_one_epoch()

with torch.no_grad():
  predicted = model(X_test.to(device)).to('cpu').numpy()

# Save Figure
output_dir = 'LSTM_test'

os.makedirs(output_dir, exist_ok=True)

for i in range(0, len(predicted), 200):
  plt.figure(figsize=(12, 9))
  plt.clf()
  plt.scatter(range(len(y_test[i:i+200,:])),y_test[i:i+200,:],label='Actual', color = 'blue', s=4)
  #plt.scatter(range(len(predicted[i:i+200,:])),predicted[i:i+200,:],label="Predicted")
  #plt.plot(y_test[i:i+200,:],label='Actual')
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
