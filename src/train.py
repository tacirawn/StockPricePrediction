"""
Training loop module for PyTorch LSTM and GRU models.
Tracks training time, loss history, and saves trained model weights.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from .data_loader import load_stock_data, scale_feature, create_sliding_windows, prepare_tensors
from .models import LSTMModel, GRUModel, count_parameters


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
    verbose: bool = True,
    print_every: int = 10
):
    """
    Train a PyTorch model on input sequences.
    
    Args:
        model: nn.Module instance (LSTMModel or GRUModel).
        x_train: Input training tensor of shape (N, seq_len, input_dim).
        y_train: Target training tensor of shape (N, output_dim).
        criterion: Loss function (e.g. nn.MSELoss).
        optimizer: PyTorch optimizer (e.g. Adam).
        num_epochs: Number of training epochs.
        verbose: Whether to print progress during training.
        print_every: Frequency of epoch logging.
        
    Returns:
        loss_history: List of MSE loss values per epoch.
        train_time: Total elapsed training time in seconds.
    """
    model.train()
    loss_history = []
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == num_epochs):
            print(f"Epoch [{epoch:3d}/{num_epochs:3d}] - Loss (MSE): {loss_val:.6f}")
            
    train_time = time.time() - start_time
    if verbose:
        print(f"--> Training completed in {train_time:.2f} seconds.")
        
    return loss_history, train_time


def run_training_experiment(
    data_path: str = "data/AMZN.csv",
    lookback: int = 20,
    hidden_dim: int = 32,
    num_layers: int = 2,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    save_dir: str = "results",
    device: str = None
):
    """
    Full training pipeline comparing LSTM and GRU on the stock dataset.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using computing device: {device}")
    
    # 1. Load and scale data
    df = load_stock_data(data_path)
    close_prices = df[['Close']].values
    scaled_data, scaler = scale_feature(close_prices, feature_range=(-1, 1))
    
    # 2. Windowing
    x_train, y_train, x_test, y_test = create_sliding_windows(scaled_data, lookback=lookback, train_ratio=0.8)
    X_train_t, y_train_t, X_test_t, y_test_t = prepare_tensors(x_train, y_train, x_test, y_test, device=device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. Train LSTM
    print("\n" + "="*50)
    print("Initializing and Training LSTM Model")
    print("="*50)
    lstm = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1).to(device)
    print(f"LSTM Parameter Count: {count_parameters(lstm)}")
    criterion = nn.MSELoss(reduction='mean')
    optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    lstm_loss, lstm_time = train_model(lstm, X_train_t, y_train_t, criterion, optimizer_lstm, num_epochs=num_epochs)
    torch.save(lstm.state_dict(), os.path.join(save_dir, "lstm_model.pth"))
    
    # 4. Train GRU
    print("\n" + "="*50)
    print("Initializing and Training GRU Model")
    print("="*50)
    gru = GRUModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1).to(device)
    print(f"GRU Parameter Count: {count_parameters(gru)}")
    optimizer_gru = torch.optim.Adam(gru.parameters(), lr=learning_rate)
    gru_loss, gru_time = train_model(gru, X_train_t, y_train_t, criterion, optimizer_gru, num_epochs=num_epochs)
    torch.save(gru.state_dict(), os.path.join(save_dir, "gru_model.pth"))
    
    return {
        "scaler": scaler,
        "lstm": {"model": lstm, "loss": lstm_loss, "time": lstm_time},
        "gru": {"model": gru, "loss": gru_loss, "time": gru_time},
        "tensors": (X_train_t, y_train_t, X_test_t, y_test_t),
        "raw_df": df
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM and GRU on Stock Price Data")
    parser.add_argument("--data", type=str, default="data/AMZN.csv", help="Path to stock CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden units dimension")
    parser.add_argument("--lookback", type=int, default=20, help="Sliding window sequence lookback length")
    args = parser.parse_args()
    
    run_training_experiment(
        data_path=args.data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        lookback=args.lookback
    )
