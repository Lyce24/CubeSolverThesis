import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
from rubik54 import FaceCube, Color  # Assuming rubik54 is a module you have access to

# Assuming color_encoding is defined as shown previously
color_encoding = {
    Color.U: np.array([1, 0, 0, 0, 0, 0]),
    Color.R: np.array([0, 1, 0, 0, 0, 0]),
    Color.F: np.array([0, 0, 1, 0, 0, 0]),
    Color.D: np.array([0, 0, 0, 1, 0, 0]),
    Color.L: np.array([0, 0, 0, 0, 1, 0]),
    Color.B: np.array([0, 0, 0, 0, 0, 1])
}

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {torch.cuda.get_device_name(0)}')

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(324, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.layers(x)

def to_rl_data(cube: FaceCube) -> np.array:
    cube_list = [color_encoding[i] for i in cube.f]
    cube_array = np.array(cube_list).reshape(1, -1)[0]
    return cube_array

def get_dataset(batch_size: int, n: int) -> tuple:
    X, Y = [], []
    for _ in range(batch_size):
        cube = FaceCube()
        k = random.randint(1, n + 1)
        cube.randomize_n(k)
        X.append(to_rl_data(cube))
        Y.append(k)
    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    Y = torch.tensor(np.array(Y), dtype=torch.float32).to(device)
    return X, Y

# Initialize the model and move it to GPU if available
model = MLP().to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Prepare dataset
States_per_update = 50000
SCRAMBLE_LENGTH = 20

# # test input
# cube = FaceCube()
# cube.randomize_n(5)
# input = to_rl_data(cube)
# print(input)

# Training loop
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
train_losses = []

model.train()  # Set the model to training mode


for epoch in range(25):
    total_loss = 0
    train_X, train_Y = get_dataset(States_per_update, SCRAMBLE_LENGTH)

    # Create DataLoaders
    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=10000, shuffle=True)
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    ax.clear()
    ax.plot(train_losses, label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Epochs')
    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()
    print(f'Epoch [{epoch+1}/100], Loss: {avg_loss:.4f}')
plt.ioff()
plt.show()

model.eval()  # Set the model to evaluation mode

i = 0
for _ in range(1000):
    cube = FaceCube()
    k = random.randint(1, SCRAMBLE_LENGTH + 1)
    cube.randomize_n(k)
    X = to_rl_data(cube)
    input = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    
    predicted_output = model(input)
    
    if abs(predicted_output.item() - k) < 0.5:
        i += 1
        
    print(f"Predicted: {predicted_output.item()}, Actual: {k}")
    
print(f"Accuracy: {i/1000}")
