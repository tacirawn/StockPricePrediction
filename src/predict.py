"""
Inference script to forecast next-day and multi-step future stock prices
using trained PyTorch LSTM or GRU checkpoints.
"""

import os
import argparse
import numpy as np
import torch
import joblib

from .models import LSTMModel, GRUModel
from .data_loader import load_stock_data, scale_feature


def predict_next_day(
    model_type: str = "gru",
    weights_path: str = "results/gru_model.pth",
    data_path: str = "data/AMZN.csv",
    lookback: int = 20,
    hidden_dim: int = 32,
    num_layers: int = 2,
    device: str = "cpu"
):
    """
    Predict the next day's closing price from the most recent historical prices.
    """
    df = load_stock_data(data_path)
    prices = df[['Close']].values
    
    scaled_data, scaler = scale_feature(prices, feature_range=(-1, 1))
    
    # Get last `lookback` days
    recent_sequence = scaled_data[-lookback:]
    x_input = torch.tensor(recent_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Load model
    if model_type.lower() == "lstm":
        model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1)
    else:
        model = GRUModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1)
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        pred_scaled = model(x_input).cpu().numpy()
        
    pred_price = scaler.inverse_transform(pred_scaled)[0, 0]
    last_actual = prices[-1, 0]
    
    print(f"\nModel: {model_type.upper()}")
    print(f"Most Recent Actual Price: ${last_actual:.2f}")
    print(f"Predicted Next-Day Price:  ${pred_price:.2f}")
    print(f"Forecasted Change:         {pred_price - last_actual:+.2f} USD ({((pred_price - last_actual)/last_actual)*100:+.2f}%)")
    
    return pred_price


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict next day stock price")
    parser.add_argument("--model", type=str, default="gru", choices=["lstm", "gru"], help="Model type")
    parser.add_argument("--weights", type=str, default="results/gru_model.pth", help="Checkpoint file")
    parser.add_argument("--data", type=str, default="data/AMZN.csv", help="Stock CSV file")
    args = parser.parse_args()
    
    predict_next_day(model_type=args.model, weights_path=args.weights, data_path=args.data)
