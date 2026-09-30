"""
Model evaluation, comparative metrics, and publication-quality visualization.
Computes RMSE, MSE, MAE, MAPE, R2 score and produces comparison charts.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate regression evaluation metrics between actual and predicted stock prices.
    
    Returns:
        dict containing MSE, RMSE, MAE, MAPE, and R2 score.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE(%)": round(mape, 2),
        "R2": round(r2, 4)
    }


def evaluate_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    scaler,
    device: str = "cpu"
):
    """
    Evaluate trained model on both train and test splits, inverting normalized scales.
    
    Returns:
        train_metrics, test_metrics, y_train_pred_inv, y_test_pred_inv
    """
    model.eval()
    with torch.no_grad():
        x_train = x_train.to(device)
        x_test = x_test.to(device)
        
        y_train_pred = model(x_train).cpu().numpy()
        y_test_pred = model(x_test).cpu().numpy()
        
    y_train_true = y_train.cpu().numpy()
    y_test_true = y_test.cpu().numpy()
    
    # Invert normalization back to original USD price scale
    y_train_pred_inv = scaler.inverse_transform(y_train_pred)
    y_train_true_inv = scaler.inverse_transform(y_train_true)
    y_test_pred_inv = scaler.inverse_transform(y_test_pred)
    y_test_true_inv = scaler.inverse_transform(y_test_true)
    
    train_metrics = compute_metrics(y_train_true_inv, y_train_pred_inv)
    test_metrics = compute_metrics(y_test_true_inv, y_test_pred_inv)
    
    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "y_train_true": y_train_true_inv,
        "y_train_pred": y_train_pred_inv,
        "y_test_true": y_test_true_inv,
        "y_test_pred": y_test_pred_inv
    }


def plot_predictions_and_loss(
    raw_df: pd.DataFrame,
    lookback: int,
    lstm_results: dict,
    gru_results: dict,
    lstm_loss: list,
    gru_loss: list,
    save_dir: str = "figures"
):
    """
    Generate beautiful comparative plots:
    1. Overall Stock Price: Actual vs LSTM vs GRU Predictions
    2. Training Loss Comparison across Epochs
    3. Zoomed Test-Set Predictions (Unseen Data Forecasting)
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    dates = pd.to_datetime(raw_df['Date']).values if 'Date' in raw_df.columns else np.arange(len(raw_df))
    actual_prices = raw_df['Close'].values
    
    train_len = len(lstm_results["y_train_pred"])
    test_len = len(lstm_results["y_test_pred"])
    
    # Map predictions to time indices
    # Sequences start from index lookback
    train_indices = np.arange(lookback - 1, lookback - 1 + train_len)
    test_indices = np.arange(lookback - 1 + train_len, lookback - 1 + train_len + test_len)
    
    # --- Figure 1: Full Historical Trajectory & Forecast ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # LSTM subplot
    axes[0].plot(dates, actual_prices, label="Actual AMZN Stock Price", color="#1f77b4", linewidth=1.8, alpha=0.8)
    axes[0].plot(dates[train_indices], lstm_results["y_train_pred"], label="LSTM Training Fit", color="#2ca02c", linestyle="--", linewidth=1.5)
    axes[0].plot(dates[test_indices], lstm_results["y_test_pred"], label="LSTM Test Prediction (Unseen)", color="#d62728", linewidth=2.0)
    axes[0].axvline(x=dates[test_indices[0]], color="gray", linestyle=":", label="Train / Test Split Point")
    axes[0].set_title("Amazon Stock Price Prediction - LSTM Model", fontsize=15, fontweight="bold", pad=10)
    axes[0].set_ylabel("Stock Price (USD)", fontsize=13)
    axes[0].legend(loc="upper left", fontsize=11)
    
    # GRU subplot
    axes[1].plot(dates, actual_prices, label="Actual AMZN Stock Price", color="#1f77b4", linewidth=1.8, alpha=0.8)
    axes[1].plot(dates[train_indices], gru_results["y_train_pred"], label="GRU Training Fit", color="#ff7f0e", linestyle="--", linewidth=1.5)
    axes[1].plot(dates[test_indices], gru_results["y_test_pred"], label="GRU Test Prediction (Unseen)", color="#9467bd", linewidth=2.0)
    axes[1].axvline(x=dates[test_indices[0]], color="gray", linestyle=":", label="Train / Test Split Point")
    axes[1].set_title("Amazon Stock Price Prediction - GRU Model", fontsize=15, fontweight="bold", pad=10)
    axes[1].set_xlabel("Date", fontsize=13)
    axes[1].set_ylabel("Stock Price (USD)", fontsize=13)
    axes[1].legend(loc="upper left", fontsize=11)
    
    plt.tight_layout()
    pred_path = os.path.join(save_dir, "actual_vs_predicted_stock_prices.png")
    plt.savefig(pred_path, dpi=300)
    plt.close()
    print(f"Saved prediction comparison plot to {pred_path}")
    
    # --- Figure 2: Zoomed-in Test Set Comparison & Training Loss ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Test Zoom
    test_dates = dates[test_indices]
    ax1.plot(test_dates, lstm_results["y_test_true"], label="Actual Price", color="black", linewidth=2.0)
    ax1.plot(test_dates, lstm_results["y_test_pred"], label=f"LSTM (Test RMSE: ${lstm_results['test_metrics']['RMSE']})", color="#d62728", linestyle="--", linewidth=1.8)
    ax1.plot(test_dates, gru_results["y_test_pred"], label=f"GRU (Test RMSE: ${gru_results['test_metrics']['RMSE']})", color="#9467bd", linestyle="-.", linewidth=1.8)
    ax1.set_title("Test Period Evaluation (Unseen Historical Sequence)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=11)
    
    # Loss curves
    epochs = range(1, len(lstm_loss) + 1)
    ax2.plot(epochs, lstm_loss, label="LSTM Loss (MSE)", color="#2ca02c", linewidth=2.0)
    ax2.plot(epochs, gru_loss, label="GRU Loss (MSE)", color="#ff7f0e", linewidth=2.0)
    ax2.set_title("Training Loss Convergence (MSE)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Mean Squared Error (Scaled)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=11)
    
    plt.tight_layout()
    test_loss_path = os.path.join(save_dir, "test_zoom_and_loss_curves.png")
    plt.savefig(test_loss_path, dpi=300)
    plt.close()
    print(f"Saved zoomed test comparison and loss curves to {test_loss_path}")
