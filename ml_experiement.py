#!/usr/bin/env python3
"""
ml_experiments.py

This script extends the ML training pipeline to:
  1. Perform a grid hyperparameter search over several training configurations.
  2. Generate a learning curve by training on increasing fractions of the data.
"""

import sys
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
from sqlalchemy.orm import sessionmaker

# Import the engine and SimulationResult from data_storage.
from digital_twin.data_storage import engine, SimulationResult

# Define the ProductionDataset (same as in your ml_trainer.py)
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ProductionDataset(Dataset):
    """
    Loads simulation data from the database.
    Input features (12 realistic parameters) are standardized.
    The target is yield_percent (a continuous value).
    """
    def __init__(self):
        # Load data from the database
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        results = session.query(SimulationResult).all()
        session.close()

        # Filter out simulation records that do not have yield_percent
        results = [r for r in results if r.yield_percent is not None]

        if not results:
            print("No simulation data with yield_percent found in the database. Please run the data collector first.")
            sys.exit(1)

        # Extract 12 realistic input features (exclude yield_percent) 
        self.features = np.array([
            [r.mixing_time, r.mixing_speed, r.uniformity_index, r.granulation_time,
             r.binder_rate, r.granule_density, r.drying_temp, r.moisture_content,
             r.comp_pressure, r.tablet_hardness, r.weight_variation, r.dissolution]
            for r in results
        ])
        # The target is yield_percent.
        self.labels = np.array([r.yield_percent for r in results])

        # Standardize features.
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

#############################################
# Function to train the model for an experiment
#############################################
def train_model_experiment(train_subset, val_loader, lr, hidden_size, batch_size, num_epochs=50):
    """
    Trains a regression model on the given training subset and evaluates on val_loader.
    
    Args:
      train_subset (Dataset): The training data (or subset) for this experiment.
      val_loader (DataLoader): The DataLoader for validation.
      lr (float): Learning rate.
      hidden_size (int): Number of neurons in the hidden layer.
      batch_size (int): Batch size for training.
      num_epochs (int): Number of training epochs.
      
    Returns:
      final_val_loss (float): The validation loss at the end of training.
    """
    # Create DataLoader for the training subset.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    input_dim = 12  # number of input features
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)  # regression output (yield_percent)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop.
    for epoch in range(num_epochs):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set.
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, targets in val_loader:
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
    final_val_loss = total_loss / total_samples
    return final_val_loss

#############################################
# Grid Hyperparameter Search
#############################################
def grid_search_experiment(dataset, full_train_set, val_loader, num_epochs=50, sample_fraction=0.5):
    """
    Perform grid search using a subset (sample_fraction) of the full training set.
    
    Args:
      dataset (Dataset): The full dataset (unused here but could be helpful for logging).
      full_train_set (Dataset): The full training set.
      val_loader (DataLoader): DataLoader for the validation set.
      num_epochs (int): Number of epochs to train.
      sample_fraction (float): Fraction of training data to use for grid search.
    
    Returns:
      best_config (dict): Best hyperparameters found.
      results (list): List of all hyperparameter configurations and their validation losses.
    """
    print("Starting grid hyperparameter search with a subset of the training data...")
    
    # Create a subset of the full training set for grid search
    num_samples = int(len(full_train_set) * sample_fraction)
    indices = list(range(num_samples))
    train_subset = torch.utils.data.Subset(full_train_set, indices)
    print(f"Using {num_samples} samples ({sample_fraction*100:.0f}%) for grid search.")

    # Define grid search hyperparameter values.
    hyperparam_grid = {
        "lr": [0.001, 0.0005, 0.0001],
        "hidden_size": [32, 64, 128],
        "batch_size": [32, 64],
    }
    
    # Create all combinations.
    grid = list(itertools.product(hyperparam_grid["lr"],
                                  hyperparam_grid["hidden_size"],
                                  hyperparam_grid["batch_size"]))
    results = []
    
    for lr, hidden_size, batch_size in grid:
        print(f"Testing configuration: lr={lr}, hidden_size={hidden_size}, batch_size={batch_size}")
        val_loss = train_model_experiment(train_subset, val_loader, lr, hidden_size, batch_size, num_epochs)
        results.append({
            "lr": lr,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "val_loss": val_loss
        })
        print(f"Validation Loss: {val_loss:.4f}")
    
    # Identify best configuration.
    best_config = min(results, key=lambda x: x["val_loss"])
    print("\nBest Hyperparameters Found:")
    print(best_config)
    return best_config, results

#############################################
# Learning Curve Generation
#############################################
def learning_curve_experiment(full_train_set, val_loader, best_params, num_epochs=100):
    print("Generating learning curve...")
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    val_losses = []
    train_sizes = []
    
    total_train = len(full_train_set)
    for frac in fractions:
        subset_size = int(total_train * frac)
        train_sizes.append(subset_size)
        # Create a subset of the training set.
        indices = list(range(subset_size))
        train_subset = Subset(full_train_set, indices)
        print(f"Training with {subset_size} samples ({frac*100:.0f}%)...")
        loss = train_model_experiment(train_subset, val_loader,
                                      lr=best_params["lr"],
                                      hidden_size=best_params["hidden_size"],
                                      batch_size=best_params["batch_size"],
                                      num_epochs=num_epochs)
        print(f"Validation Loss with {subset_size} samples: {loss:.4f}")
        val_losses.append(loss)
    
    # Plot the learning curve.
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, val_losses, marker='o')
    plt.title("Learning Curve")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Validation Loss (MSE)")
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.show()

#############################################
# Main experimental routine
#############################################
def main():
    # Load the full dataset.
    dataset = ProductionDataset()
    total_size = len(dataset)
    print(f"Total simulation records: {total_size}")

    # Split the dataset into training (80%) and validation (20%).
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    full_train_set, val_set = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {len(full_train_set)}, Validation set size: {len(val_set)}")

    # Create a validation DataLoader.
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # --- Part 1: Grid Hyperparameter Search ---
    best_params, grid_results = grid_search_experiment(dataset, full_train_set, val_loader, num_epochs=50)

    # --- Part 2: Learning Curve Generation ---
    # Use the best hyperparameters from the grid search.
    learning_curve_experiment(full_train_set, val_loader, best_params, num_epochs=100)

if __name__ == "__main__":
    main()
