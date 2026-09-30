"""
Data preprocessing, scaling, and sequence windowing for Time Series Stock Prediction.
Implements the sliding-window sequence approach as described in the benchmark plan.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    """PyTorch Dataset for time series sliding windows."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_stock_data(filepath: str, sort_by_date: bool = True) -> pd.DataFrame:
    """
    Load stock time series data from CSV file.
    
    Args:
        filepath: Path to the CSV file.
        sort_by_date: Whether to sort by Date column ascending.
        
    Returns:
        Pandas DataFrame containing stock market records.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Stock data file not found: {filepath}")

    data = pd.read_csv(filepath)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        if sort_by_date:
            data = data.sort_values('Date').reset_index(drop=True)
    return data


def scale_feature(series: np.ndarray, feature_range: tuple = (-1, 1)):
    """
    Scale numeric series using MinMaxScaler.
    
    Args:
        series: 1D or 2D array of prices.
        feature_range: Normalization min and max bounds. Default is (-1, 1).
        
    Returns:
        scaled_data: Scaled numpy array.
        scaler: Fitted MinMaxScaler instance.
    """
    if len(series.shape) == 1:
        series = series.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_data = scaler.fit_transform(series)
    return scaled_data, scaler


def create_sliding_windows(
    data: np.ndarray,
    lookback: int = 20,
    train_ratio: float = 0.8
):
    """
    Generate sliding-window sequences for time-series regression.
    
    Given a sequence of length L, for index i:
        Input sequence X[i]  = data[i : i + lookback]
        Target value  y[i]  = data[i + lookback] (next day's price)
        
    Splits into train and test sets strictly preserving chronological order.
    
    Args:
        data: Scaled time series array of shape (N, num_features).
        lookback: Number of past time steps to observe (window size).
        train_ratio: Proportion of sequences for training (default 0.8).
        
    Returns:
        x_train, y_train, x_test, y_test as NumPy arrays.
    """
    sequences = []
    
    # Create all possible sequences of length lookback
    for index in range(len(data) - lookback):
        sequences.append(data[index : index + lookback])
        
    sequences = np.array(sequences)
    
    total_samples = sequences.shape[0]
    test_set_size = int(np.round((1.0 - train_ratio) * total_samples))
    train_set_size = total_samples - test_set_size
    
    # Input sequences (all time steps except the last if targets are included,
    # or lookback steps as input and the (lookback)-th step as target)
    x_train = sequences[:train_set_size, :-1, :]
    y_train = sequences[:train_set_size, -1, :]
    
    x_test = sequences[train_set_size:, :-1, :]
    y_test = sequences[train_set_size:, -1, :]
    
    return x_train, y_train, x_test, y_test


def prepare_tensors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu"
):
    """
    Convert NumPy arrays to PyTorch Tensors and send to specified device.
    """
    X_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_test_t = torch.from_numpy(x_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)
    
    return X_train_t, y_train_t, X_test_t, y_test_t
