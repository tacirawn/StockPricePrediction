"""
Flask Web Application & REST API
Stock Price Prediction with PyTorch (LSTM vs GRU)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, jsonify, request

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import LSTMModel, GRUModel, count_parameters
from src.data_loader import load_stock_data, scale_feature, create_sliding_windows, prepare_tensors

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "web", "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "web", "static")
)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def add_no_cache_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "AMZN.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LSTM_WEIGHTS = os.path.join(RESULTS_DIR, "lstm_model.pth")
GRU_WEIGHTS = os.path.join(RESULTS_DIR, "gru_model.pth")

# Global cached structures
DATA_CACHE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKBACK = 20
HIDDEN_DIM = 32
NUM_LAYERS = 2


def get_preprocessed_cache():
    global DATA_CACHE
    if DATA_CACHE is not None:
        return DATA_CACHE

    df = load_stock_data(DATA_PATH)
    prices = df[['Close']].values
    scaled_data, scaler = scale_feature(prices, feature_range=(-1, 1))

    x_train, y_train, x_test, y_test = create_sliding_windows(
        scaled_data, lookback=LOOKBACK, train_ratio=0.8
    )
    X_train_t, y_train_t, X_test_t, y_test_t = prepare_tensors(
        x_train, y_train, x_test, y_test, device=DEVICE
    )

    # Initialize models
    lstm = LSTMModel(input_dim=1, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1).to(DEVICE)
    gru = GRUModel(input_dim=1, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1).to(DEVICE)

    if os.path.exists(LSTM_WEIGHTS):
        lstm.load_state_dict(torch.load(LSTM_WEIGHTS, map_location=DEVICE))
    if os.path.exists(GRU_WEIGHTS):
        gru.load_state_dict(torch.load(GRU_WEIGHTS, map_location=DEVICE))

    lstm.eval()
    gru.eval()

    # Pre-generate inference
    with torch.no_grad():
        y_train_lstm_pred = scaler.inverse_transform(lstm(X_train_t).cpu().numpy())
        y_test_lstm_pred = scaler.inverse_transform(lstm(X_test_t).cpu().numpy())
        y_train_gru_pred = scaler.inverse_transform(gru(X_train_t).cpu().numpy())
        y_test_gru_pred = scaler.inverse_transform(gru(X_test_t).cpu().numpy())

    DATA_CACHE = {
        "df": df,
        "prices": prices,
        "scaled_data": scaled_data,
        "scaler": scaler,
        "lstm": lstm,
        "gru": gru,
        "tensors": (X_train_t, y_train_t, X_test_t, y_test_t),
        "y_train_lstm_pred": y_train_lstm_pred,
        "y_test_lstm_pred": y_test_lstm_pred,
        "y_train_gru_pred": y_train_gru_pred,
        "y_test_gru_pred": y_test_gru_pred,
        "train_len": len(y_train_lstm_pred),
        "test_len": len(y_test_lstm_pred)
    }
    return DATA_CACHE


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    cache = get_preprocessed_cache()
    df = cache["df"]
    prices = cache["prices"]
    
    first_date = str(df['Date'].iloc[0].strftime('%Y-%m-%d'))
    last_date = str(df['Date'].iloc[-1].strftime('%Y-%m-%d'))
    last_price = float(prices[-1, 0])
    first_price = float(prices[0, 0])
    
    lstm_params = count_parameters(cache["lstm"])
    gru_params = count_parameters(cache["gru"])
    
    return jsonify({
        "ticker": "AMZN",
        "company": "Amazon.com, Inc.",
        "startDate": first_date,
        "endDate": last_date,
        "totalDays": len(df),
        "startPrice": round(first_price, 2),
        "lastPrice": round(last_price, 2),
        "overallGrowth": f"{((last_price - first_price) / first_price) * 100:+.1f}%",
        "device": DEVICE.upper(),
        "lstmParams": lstm_params,
        "gruParams": gru_params,
        "paramReduction": f"{(1 - gru_params / lstm_params) * 100:.1f}%",
        "lookbackWindow": LOOKBACK
    })


@app.route("/api/chart-data")
def api_chart_data():
    cache = get_preprocessed_cache()
    df = cache["df"]
    
    dates = [d.strftime('%Y-%m-%d') for d in pd.to_datetime(df['Date'])]
    actual_prices = [round(float(p), 2) for p in df['Close'].values]
    
    train_len = cache["train_len"]
    test_len = cache["test_len"]
    
    # Initialize arrays with None
    lstm_train = [None] * len(df)
    lstm_test = [None] * len(df)
    gru_train = [None] * len(df)
    gru_test = [None] * len(df)
    
    # Fill in predictions starting at LOOKBACK - 1
    start_train = LOOKBACK - 1
    for i in range(train_len):
        idx = start_train + i
        if idx < len(df):
            lstm_train[idx] = round(float(cache["y_train_lstm_pred"][i, 0]), 2)
            gru_train[idx] = round(float(cache["y_train_gru_pred"][i, 0]), 2)
            
    start_test = start_train + train_len
    for i in range(test_len):
        idx = start_test + i
        if idx < len(df):
            lstm_test[idx] = round(float(cache["y_test_lstm_pred"][i, 0]), 2)
            gru_test[idx] = round(float(cache["y_test_gru_pred"][i, 0]), 2)
            
    split_date = dates[start_test] if start_test < len(dates) else dates[-1]
    
    # Downsample option if requested for performance
    step = int(request.args.get("step", 1))
    if step > 1:
        dates = dates[::step]
        actual_prices = actual_prices[::step]
        lstm_train = lstm_train[::step]
        lstm_test = lstm_test[::step]
        gru_train = gru_train[::step]
        gru_test = gru_test[::step]
        
    return jsonify({
        "dates": dates,
        "actual": actual_prices,
        "lstmTrain": lstm_train,
        "lstmTest": lstm_test,
        "gruTrain": gru_train,
        "gruTest": gru_test,
        "splitDate": split_date
    })


@app.route("/api/metrics")
def api_metrics():
    comp_file = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_file):
        df_comp = pd.read_csv(comp_file)
        data = df_comp.to_dict(orient="records")
    else:
        data = []
    return jsonify(data)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    cache = get_preprocessed_cache()
    scaler = cache["scaler"]
    prices = cache["prices"]
    
    req_data = request.get_json(silent=True) or {}
    custom_sequence = req_data.get("sequence", None)
    
    if custom_sequence and len(custom_sequence) >= LOOKBACK:
        input_prices = np.array(custom_sequence[-LOOKBACK:]).reshape(-1, 1)
        scaled_input = scaler.transform(input_prices)
    else:
        # Use latest 20 actual days from dataset
        scaled_input = cache["scaled_data"][-LOOKBACK:]
        input_prices = prices[-LOOKBACK:]
        
    x_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        lstm_pred_scaled = cache["lstm"](x_tensor).cpu().numpy()
        gru_pred_scaled = cache["gru"](x_tensor).cpu().numpy()
    inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    lstm_val = round(float(scaler.inverse_transform(lstm_pred_scaled)[0, 0]), 2)
    gru_val = round(float(scaler.inverse_transform(gru_pred_scaled)[0, 0]), 2)
    last_val = round(float(input_prices[-1, 0]), 2)
    
    return jsonify({
        "currentPrice": last_val,
        "lstm": {
            "predictedPrice": lstm_val,
            "change": round(lstm_val - last_val, 2),
            "pctChange": round(((lstm_val - last_val) / last_val) * 100, 2),
            "recommendation": "AL" if lstm_val > last_val else "SAT"
        },
        "gru": {
            "predictedPrice": gru_val,
            "change": round(gru_val - last_val, 2),
            "pctChange": round(((gru_val - last_val) / last_val) * 100, 2),
            "recommendation": "AL" if gru_val > last_val else "SAT"
        },
        "latencyMs": inference_time_ms,
        "device": DEVICE.upper(),
        "recentDays": [round(float(p[0]), 2) for p in input_prices]
    })


if __name__ == "__main__":
    print(f"Starting Stock Prediction Web Server on http://localhost:5000 (Device: {DEVICE.upper()})")
    app.run(host="0.0.0.0", port=5000, debug=False)
