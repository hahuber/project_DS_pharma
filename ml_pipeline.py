#!/usr/bin/env python3
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import sessionmaker

# Import the SQLAlchemy engine and SimulationResult ORM model from our data_storage module.
from digital_twin.data_storage import engine, SimulationResult

########################################
# Step 1: Check if Data Is Present in DB
########################################
def check_data():
    """Ensure that the database contains simulation data."""
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    record_count = session.query(SimulationResult).count()
    session.close()
    if record_count == 0:
        print("No simulation data found in the database. Please run the data collector first.")
        sys.exit(1)
    else:
        print(f"Found {record_count} simulation records in the database.")

########################################
# Step 2: Define the PyTorch Dataset
########################################
class ProductionDataset(Dataset):
    """
    Loads simulation data from the database,
    extracts features and labels, and applies standardization.
    """
    def __init__(self):
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        results = session.query(SimulationResult).all()
        session.close()

        # Build the feature array.
        self.features = np.array([
            [r.mixing_time, r.mixing_speed, r.uniformity_index, r.granulation_time,
             r.binder_rate, r.granule_density, r.drying_temp, r.moisture_content,
             r.comp_pressure, r.tablet_hardness, r.weight_variation, r.dissolution, r.yield_percent]
            for r in results
        ])

        # Build the label array. We assume that 'final_status' is "success" or "failure".
        self.labels = np.array([1 if r.final_status == "success" else 0 for r in results])

        # Standardize the features.
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return a tuple of (features, label) as tensors.
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

########################################
# Step 3: Define the Model Evaluation Function
########################################
def evaluate_model(model, loader, criterion):
    """
    Evaluates the model on data from a given DataLoader.
    Returns the average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for features, targets in loader:
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item() * features.size(0)
            # Binary classification: threshold predictions at 0.5.
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == targets).sum().item()
            total_samples += features.size(0)
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

########################################
# Step 4: Train and Evaluate the Model
########################################
def train_model():
    # Check that simulation data exists.
    check_data()

    # Load the dataset from the database.
    dataset = ProductionDataset()
    total_size = len(dataset)
    print(f"Total dataset size: {total_size}")

    # Split the dataset: 80% for training and 20% for validation.
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {len(train_set)}, Validation set size: {len(val_set)}")

    # Create DataLoaders.
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Define a simple neural network.
    model = nn.Sequential(
        nn.Linear(13, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * features.size(0)
        
        avg_train_loss = total_train_loss / len(train_set)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    # Save the trained model.
    torch.save(model.state_dict(), "digital_twin/production_model.pth")
    print("Model training complete and saved as 'digital_twin/production_model.pth'.")

########################################
# Main entry point
########################################
if __name__ == "__main__":
    train_model()

