import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sqlalchemy.orm import sessionmaker
from digital_twin.data_storage import engine, SimulationResult
from sklearn.preprocessing import StandardScaler

class ProductionDataset(Dataset):
    """
    PyTorch Dataset that loads simulation data from the database.
    The features are the measurable process parameters (excluding yield_percent),
    and the target is the yield_percent.
    """
    def __init__(self):
        # Load data from the database
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        results = session.query(SimulationResult).all()
        session.close()

        if not results:
            print("No simulation data found in the database. Please run the data collector first.")
            sys.exit(1)

        # Define the input features that are realistically measurable.
        # We exclude 'yield_percent' since that is our prediction target.
        self.features = np.array([
            [r.mixing_time, r.mixing_speed, r.uniformity_index, r.granulation_time,
             r.binder_rate, r.granule_density, r.drying_temp, r.moisture_content,
             r.comp_pressure, r.tablet_hardness, r.weight_variation, r.dissolution]
            for r in results
        ])

        # The target is the yield_percent value.
        self.labels = np.array([r.yield_percent for r in results])

        # Standardize features to help the training process.
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return features and label as PyTorch tensors.
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

def evaluate_model(model, loader, criterion):
    """
    Evaluate the model on a given DataLoader.
    Returns the average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for features, targets in loader:
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item() * features.size(0)
            count += features.size(0)
    return total_loss / count

def train_model():
    # Load the dataset
    dataset = ProductionDataset()
    total_size = len(dataset)
    print(f"Total simulation records: {total_size}")

    # Split the dataset: 80% training, 20% validation.
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {len(train_set)}, Validation set size: {len(val_set)}")

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Define the regression model.
    # Input dimension is 12 (the number of realistic features), and the output is a single continuous value.
    model = nn.Sequential(
        nn.Linear(12, 64),
        nn.ReLU(),
        nn.Linear(64, 1)  # No activation function, since we predict a continuous value.
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # For regression tasks

    num_epochs = 50
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

        avg_train_loss = total_train_loss / train_size
        val_loss = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "digital_twin/production_model.pth")
    print("Model training complete and saved as 'digital_twin/production_model.pth'.")

if __name__ == "__main__":

    train_model()
