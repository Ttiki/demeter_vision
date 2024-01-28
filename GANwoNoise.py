import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
import ot
# Step 1: Data Loading and Preprocessing
# Load the data
noise = np.load("noise.npy")
station_40 = pd.read_csv('station_40.csv')
station_49 = pd.read_csv('station_49.csv')
station_63 = pd.read_csv('station_63.csv')
station_80 = pd.read_csv('station_80.csv')

# Combine data from all stations (assuming same order and number of records)
combined_data = pd.concat([station_40, station_49, station_63, station_80], axis=1)

# Create the subset E based on the given conditions
Q1, Q2, Q3, Q4 = 3.3241, 5.1292, 6.4897, 7.1301  # Replace with actual quantile values if different
subset_E = combined_data[
    (station_49['W_13'] + station_49['W_14'] + station_49['W_15'] <= Q1) &
    (station_80['W_13'] + station_80['W_14'] + station_80['W_15'] <= Q2) &
    (station_40['W_13'] + station_40['W_14'] + station_40['W_15'] <= Q3) &
    (station_63['W_13'] + station_63['W_14'] + station_63['W_15'] <= Q4)
]

# Extract features (W) and targets (Y) for subset E
features = subset_E.iloc[:, :-1].values  # Assuming last 4 columns are yields
targets = subset_E.iloc[:, -1:].values   # Assuming last 4 columns are yields

# Normalize features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Convert to PyTorch tensors
tensor_x = torch.Tensor(features_normalized)
tensor_y = torch.Tensor(targets)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(tensor_x, tensor_y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 2: Model Definition
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, z):
        return self.network(z)

# Instantiate the model
input_dim = features.shape[1]
output_dim = targets.shape[1]
generator = Generator(input_dim, output_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.001)

# Step 3: Training Loop
num_epochs = 150
for epoch in range(num_epochs):
    total_swd =0
    for batch_x, batch_y in dataloader:
        # Forward pass
        gen_y = generator(batch_x)
        loss = criterion(gen_y, batch_y)
        # Compute Sliced Wasserstein Distance
        # Convert tensors to numpy for SWD computation
        synthetic_yields_np = gen_y.detach().cpu().numpy()
        real_yields_np = batch_y.detach().cpu().numpy()
        swd = ot.sliced.sliced_wasserstein_distance(real_yields_np, synthetic_yields_np, seed=0)
        total_swd += swd

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every few epochs
    average_swd = total_swd / len(dataloader)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, Average SWD: {average_swd:.4f}')


