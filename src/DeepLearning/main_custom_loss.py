import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lossFunction import DiceLoss, DiceBCELoss, IoULoss, FocalLoss, TverskyLoss, FocalTverskyLoss, ComboLoss
from PreProcessing import pre_processing

class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, dataloader, epochs):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)

# Load Data
data_path = 'Advanced-DA-Task-management.csv'
data = pd.read_csv(data_path)
X, y = pre_processing(data, pytorch=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = X_train.shape[1]
hidden_size = 100
output_size = 1

# Thay loss function
# Apply custom Loss Function
# loss_fn = FocalLoss()
# loss_fn = IoULoss()
# loss_fn = DiceBCELoss()
# loss_fn = TverskyLoss()
# loss_fn = FocalTverskyLoss()
loss_fn = ComboLoss()

model = CustomMLP(input_size, hidden_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation
trainer = Trainer(model, loss_fn, optimizer)
trainer.train(train_loader, epochs=10)
print("Test Loss:", trainer.evaluate(test_loader))


"""
Bổ sung thêm 1 file chứa all custom loss function
1 file tạo custom MLP dùng pytorch
Các loss function đều fit được
Kết quả của rất tệ, 
có ComboLoss còn bị âm và inf
"""