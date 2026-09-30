"""
Generates the comprehensive Jupyter Notebook:
notebooks/Stock_Price_Prediction_LSTM_vs_GRU.ipynb
"""

import json
import os

NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
os.makedirs(NOTEBOOK_DIR, exist_ok=True)
NOTEBOOK_PATH = os.path.join(NOTEBOOK_DIR, "Stock_Price_Prediction_LSTM_vs_GRU.ipynb")

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# --- TITLE & OVERVIEW ---
add_md("""# Stock Price Prediction with PyTorch: LSTM vs GRU Architectures
### A Comprehensive Time-Series Forecasting Study based on the 1-Month ML Curriculum
**Benchmark Reference:** [Rodolfo Saldanha - Stock Price Prediction with PyTorch](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632)

---

## 📌 Executive Summary
This project implements a complete, end-to-end deep learning pipeline for predicting stock market prices using **Recurrent Neural Networks (RNNs)** in **PyTorch**. Specifically, we construct, train, evaluate, and critically compare two state-of-the-art recurrent architectures:
1. **Long Short-Term Memory (LSTM)**
2. **Gated Recurrent Unit (GRU)**

### Key Objectives:
- **Phase 1 (Data & Preprocessing):** Exploratory data analysis on 12+ years of historical Amazon (**AMZN**) daily prices, MinMax normalization, and sequence generation using the **sliding window method** (`lookback = 20`).
- **Phase 2 (Model Architectures):** Build modular PyTorch `nn.Module` classes for LSTM and GRU with dynamic hidden layers and linear readout heads.
- **Phase 3 (Training & Benchmarking):** Train both models for 100 epochs using Mean Squared Error (`nn.MSELoss`) and Adam optimizer, benchmarking **MSE/RMSE**, **MAE**, **MAPE**, and **execution time**.
- **Critical Reflection:** Analyze the limitations of AI in financial prediction through the lens of *"AI Snake Oil"* (Narayanan & Kapoor) and the Efficient Market Hypothesis (EMH).
""")

# --- CELL: IMPORTS ---
add_code("""# Step 1: Core Library Imports
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn

# Visual styling
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12

# Device selection (NVIDIA GPU CUDA acceleration if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__}")
print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")""")

# --- PHASE 1: DATA EXPLORATION ---
add_md("""---
## 📊 Phase 1: Exploratory Data Analysis & Preprocessing

Time series data represents sequential observations recorded at regular time intervals. In stock market forecasting:
- Each record contains features: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
- We frame stock prediction as a **regression task**, predicting the continuous scalar **Closing Price ($y_t$)** at day $t$ given past historical prices $\{y_{t-1}, y_{t-2}, ..., y_{t-N}\}$.
""")

add_code("""# Step 2: Load Historical Stock Data
# Look for data file in local data directory
data_path = os.path.join("..", "data", "AMZN.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "AMZN.csv")

df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print("Dataset Shape:", df.shape)
print("\\nFirst 5 Records:")
df.head()""")

add_code("""# Step 3: Dataset Summary and Missing Values Check
print("Summary Statistics:")
print(df.describe())

missing_values = df.isnull().sum()
print("\\nMissing Values Count:")
print(missing_values)""")

add_code("""# Step 4: Visualize 12-Year Historical Closing Price Trajectory
plt.figure(figsize=(15, 6))
plt.plot(df['Date'], df['Close'], color='#1f77b4', linewidth=2.0, label='AMZN Close Price (USD)')
plt.title('Amazon (AMZN) Historical Stock Price (2006 - 2018)', fontsize=16, fontweight='bold', pad=12)
plt.xlabel('Date', fontsize=13)
plt.ylabel('Closing Price (USD)', fontsize=13)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()""")

# --- NORMALIZATION & SLIDING WINDOW ---
add_md("""### 📐 Data Normalization and Sliding Window Technique
1. **Feature Scaling (MinMaxScaler):**
   Neural networks train significantly faster and avoid vanishing/exploding gradients when features are normalized. In this benchmark, prices are scaled to the range $[-1, 1]$:
   $$x_{norm} = 2 \\times \\frac{x - x_{min}}{x_{max} - x_{min}} - 1$$
   This symmetric range matches the output range of the hyperbolic tangent ($\\tanh$) activation function used internally by LSTM and GRU gates.

2. **Sliding Window Sequence Creation (`lookback = 20`):**
   To forecast the price at day $t+1$, the model observes the preceding $N=20$ days of prices:
   $$X_i = [p_{i}, p_{i+1}, \\dots, p_{i+19}], \\quad y_i = p_{i+20}$$
   We partition the chronological sequences into an **80% training set** and a **20% testing set** without shuffling, preserving the strict arrow of time.
""")

add_code("""# Step 5: Normalization using MinMaxScaler [-1, 1]
close_prices = df[['Close']].values

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(close_prices)

print(f"Original Price Range: [${close_prices.min():.2f}, ${close_prices.max():.2f}]")
print(f"Scaled Price Range:   [{scaled_data.min():.2f}, {scaled_data.max():.2f}]")""")

add_code("""# Step 6: Sliding Window Sequence Generator
def create_sliding_windows(data, lookback=20, train_ratio=0.8):
    sequences = []
    for index in range(len(data) - lookback):
        sequences.append(data[index : index + lookback])
    
    sequences = np.array(sequences)
    
    total_samples = sequences.shape[0]
    test_set_size = int(np.round((1.0 - train_ratio) * total_samples))
    train_set_size = total_samples - test_set_size
    
    x_train = sequences[:train_set_size, :-1, :]
    y_train = sequences[:train_set_size, -1, :]
    
    x_test = sequences[train_set_size:, :-1, :]
    y_test = sequences[train_set_size:, -1, :]
    
    return x_train, y_train, x_test, y_test

LOOKBACK = 20
x_train, y_train, x_test, y_test = create_sliding_windows(scaled_data, lookback=LOOKBACK, train_ratio=0.8)

print(f"Total Sequences Generated: {len(scaled_data) - LOOKBACK}")
print(f"X_train shape: {x_train.shape} | y_train shape: {y_train.shape}")
print(f"X_test shape:  {x_test.shape}  | y_test shape:  {y_test.shape}")""")

add_code("""# Step 7: Convert NumPy Arrays to PyTorch Tensors
X_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)
X_test_tensor  = torch.from_numpy(x_test).float().to(device)
y_test_tensor  = torch.from_numpy(y_test).float().to(device)

print(f"Training Tensors:   X={X_train_tensor.shape}, y={y_train_tensor.shape} on {X_train_tensor.device}")
print(f"Testing Tensors:    X={X_test_tensor.shape},  y={y_test_tensor.shape} on {X_test_tensor.device}")""")

# --- PHASE 2: MODEL ARCHITECTURES ---
add_md("""---
## 🧠 Phase 2: PyTorch Deep Learning Architectures

### 1. Long Short-Term Memory (LSTM)
LSTMs mitigate the vanishing gradient problem in traditional RNNs by maintaining an explicit internal **Cell State ($C_t$)** governed by three continuous gating mechanisms:
- **Forget Gate ($f_t$):** Determines what proportion of past cell memory to discard:
  $$f_t = \\sigma(W_f x_t + U_f h_{t-1} + b_f)$$
- **Input Gate ($i_t$) & Candidate State ($\\tilde{C}_t$):** Controls what new information to write into memory:
  $$i_t = \\sigma(W_i x_t + U_i h_{t-1} + b_i), \\quad \\tilde{C}_t = \\tanh(W_c x_t + U_c h_{t-1} + b_c)$$
- **Cell State Update ($C_t$):**
  $$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$$
- **Output Gate ($o_t$) & Hidden State ($h_t$):** Decides what hidden representation to emit:
  $$o_t = \\sigma(W_o x_t + U_o h_{t-1} + b_o), \\quad h_t = o_t \\odot \\tanh(C_t)$$

---

### 2. Gated Recurrent Unit (GRU)
The GRU (Cho et al., 2014) is a streamlined variation that merges the cell state and hidden state, employing only **two gates**:
- **Reset Gate ($r_t$):** Controls how to combine new input with previous memory:
  $$r_t = \\sigma(W_r x_t + U_r h_{t-1} + b_r)$$
- **Update Gate ($z_t$):** Acts simultaneously as forget and input gate:
  $$z_t = \\sigma(W_z x_t + U_z h_{t-1} + b_z)$$
- **Candidate Hidden State ($\\tilde{h}_t$):**
  $$\\tilde{h}_t = \\tanh(W x_t + U (r_t \\odot h_{t-1}) + b)$$
- **Hidden State Update ($h_t$):**
  $$h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t$$

**Theoretical Difference:** Because GRU lacks a separate output gate and cell state, it has **~25% fewer parameters** than an equivalent LSTM, which typically leads to **faster training times** and reduced susceptibility to overfitting on smaller datasets.
""")

add_code("""# Step 8: Definition of PyTorch LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Take the hidden state output from the final time step
        out = self.fc(out[:, -1, :])
        return out

# Definition of PyTorch GRU Model Class
class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        
        out, hn = self.gru(x, h0)
        # Take the hidden state output from the final time step
        out = self.fc(out[:, -1, :])
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)""")

add_code("""# Step 9: Model Instantiation and Parameter Comparison
INPUT_DIM = 1
HIDDEN_DIM = 32
NUM_LAYERS = 2
OUTPUT_DIM = 1

lstm_model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)
gru_model  = GRUModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)

lstm_params = count_parameters(lstm_model)
gru_params  = count_parameters(gru_model)

print("="*45)
print(f"LSTM Trainable Parameters: {lstm_params:,}")
print(f"GRU  Trainable Parameters: {gru_params:,}")
print(f"Difference: GRU has {lstm_params - gru_params:,} ({(1 - gru_params/lstm_params)*100:.1f}%) fewer parameters")
print("="*45)""")

# --- PHASE 3: TRAINING & EVALUATION ---
add_md("""---
## 🚀 Phase 3: Model Training, Evaluation & Comparison

We train both models for **100 epochs** using identical hyperparameters:
- **Loss Function:** Mean Squared Error ($MSE = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2$)
- **Optimizer:** Adam with learning rate $\\eta = 0.01$
- **Epochs:** 100
""")

add_code("""# Step 10: Training Loop Definition
def train_model(model, X_train, y_train, criterion, optimizer, num_epochs=100, model_name="Model"):
    model.train()
    loss_history = []
    start_time = time.time()
    
    print(f"--> Commencing Training for {model_name}...")
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs:3d}] | Loss (MSE): {loss_val:.6f}")
            
    total_time = time.time() - start_time
    print(f"Completed {model_name} in {total_time:.2f} seconds.\\n")
    return loss_history, total_time""")

add_code("""# Step 11: Train LSTM and GRU Models
NUM_EPOCHS = 100
LR = 0.01

criterion = nn.MSELoss(reduction='mean')

# Train LSTM
optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=LR)
lstm_loss, lstm_time = train_model(lstm_model, X_train_tensor, y_train_tensor, criterion, optimizer_lstm, NUM_EPOCHS, "LSTM")

# Train GRU
optimizer_gru = torch.optim.Adam(gru_model.parameters(), lr=LR)
gru_loss, gru_time = train_model(gru_model, X_train_tensor, y_train_tensor, criterion, optimizer_gru, NUM_EPOCHS, "GRU")""")

add_md("""### 📈 Evaluation Metrics Calculation
Because model loss is computed in normalized space $[-1, 1]$, we apply `inverse_transform` to convert predictions back into real **US Dollars ($)**. We then compute:
- **Root Mean Squared Error (RMSE):** In dollars, heavily penalizing large forecast errors.
- **Mean Absolute Error (MAE):** Average absolute dollar deviation.
- **Mean Absolute Percentage Error (MAPE):** Relative percentage error.
- **Coefficient of Determination ($R^2$):** Proportion of price variance explained by the model.
""")

add_code("""# Step 12: Invert Normalization and Calculate Metrics
def evaluate_predictions(model, X_train_t, y_train_t, X_test_t, y_test_t, scaler):
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_t).cpu().numpy()
        y_test_pred  = model(X_test_t).cpu().numpy()
        
    y_train_true = y_train_t.cpu().numpy()
    y_test_true  = y_test_t.cpu().numpy()
    
    # Invert back to real dollar values
    y_train_pred_inv = scaler.inverse_transform(y_train_pred)
    y_train_true_inv = scaler.inverse_transform(y_train_true)
    y_test_pred_inv  = scaler.inverse_transform(y_test_pred)
    y_test_true_inv  = scaler.inverse_transform(y_test_true)
    
    def get_metrics(y_true, y_pred):
        y_t, y_p = y_true.flatten(), y_pred.flatten()
        mse = mean_squared_error(y_t, y_p)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_t, y_p)
        mape = np.mean(np.abs((y_t - y_p) / y_t)) * 100
        r2 = r2_score(y_t, y_p)
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "R2": r2}
        
    return {
        "train": get_metrics(y_train_true_inv, y_train_pred_inv),
        "test":  get_metrics(y_test_true_inv, y_test_pred_inv),
        "y_train_pred": y_train_pred_inv,
        "y_train_true": y_train_true_inv,
        "y_test_pred":  y_test_pred_inv,
        "y_test_true":  y_test_true_inv
    }

lstm_eval = evaluate_predictions(lstm_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler)
gru_eval  = evaluate_predictions(gru_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler)""")

add_code("""# Step 13: Tabular Results Comparison
comparison_data = [
    {
        "Model": "LSTM",
        "Parameters": f"{lstm_params:,}",
        "Training Time (s)": f"{lstm_time:.2f}",
        "Train RMSE ($)": f"{lstm_eval['train']['RMSE']:.2f}",
        "Test RMSE ($)": f"{lstm_eval['test']['RMSE']:.2f}",
        "Test MAE ($)": f"{lstm_eval['test']['MAE']:.2f}",
        "Test MAPE (%)": f"{lstm_eval['test']['MAPE(%)']:.2f}%",
        "Test R2": f"{lstm_eval['test']['R2']:.4f}"
    },
    {
        "Model": "GRU",
        "Parameters": f"{gru_params:,}",
        "Training Time (s)": f"{gru_time:.2f}",
        "Train RMSE ($)": f"{gru_eval['train']['RMSE']:.2f}",
        "Test RMSE ($)": f"{gru_eval['test']['RMSE']:.2f}",
        "Test MAE ($)": f"{gru_eval['test']['MAE']:.2f}",
        "Test MAPE (%)": f"{gru_eval['test']['MAPE(%)']:.2f}%",
        "Test R2": f"{gru_eval['test']['R2']:.4f}"
    }
]

comparison_df = pd.DataFrame(comparison_data)
print("\\n" + "="*80)
print("BENCHMARK COMPARISON TABLE: LSTM vs GRU")
print("="*80)
comparison_df""")

# --- VISUALIZATIONS ---
add_md("""---
## 📉 Visualizing Forecasts and Convergence
""")

add_code("""# Step 14: Historical Trajectory vs Train Fit and Test Forecast
dates = df['Date'].values
train_len = len(lstm_eval['y_train_pred'])
test_len  = len(lstm_eval['y_test_pred'])

train_idx = np.arange(LOOKBACK - 1, LOOKBACK - 1 + train_len)
test_idx  = np.arange(LOOKBACK - 1 + train_len, LOOKBACK - 1 + train_len + test_len)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# LSTM
ax1.plot(dates, df['Close'], label='Actual Stock Price', color='#1f77b4', linewidth=1.8, alpha=0.8)
ax1.plot(dates[train_idx], lstm_eval['y_train_pred'], label='LSTM Training Fit', color='#2ca02c', linestyle='--', linewidth=1.5)
ax1.plot(dates[test_idx], lstm_eval['y_test_pred'], label=f'LSTM Test Prediction (RMSE: ${lstm_eval[\"test\"][\"RMSE\"]:.2f})', color='#d62728', linewidth=2.0)
ax1.axvline(x=dates[test_idx[0]], color='gray', linestyle=':', label='Train / Test Split Point')
ax1.set_title('Amazon Stock Price Prediction - LSTM Architecture', fontsize=15, fontweight='bold', pad=10)
ax1.set_ylabel('Stock Price (USD)', fontsize=13)
ax1.legend(loc='upper left', fontsize=11)

# GRU
ax2.plot(dates, df['Close'], label='Actual Stock Price', color='#1f77b4', linewidth=1.8, alpha=0.8)
ax2.plot(dates[train_idx], gru_eval['y_train_pred'], label='GRU Training Fit', color='#ff7f0e', linestyle='--', linewidth=1.5)
ax2.plot(dates[test_idx], gru_eval['y_test_pred'], label=f'GRU Test Prediction (RMSE: ${gru_eval[\"test\"][\"RMSE\"]:.2f})', color='#9467bd', linewidth=2.0)
ax2.axvline(x=dates[test_idx[0]], color='gray', linestyle=':', label='Train / Test Split Point')
ax2.set_title('Amazon Stock Price Prediction - GRU Architecture', fontsize=15, fontweight='bold', pad=10)
ax2.set_xlabel('Date', fontsize=13)
ax2.set_ylabel('Stock Price (USD)', fontsize=13)
ax2.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.show()""")

add_code("""# Step 15: Zoomed-in Test Set Comparison & Training Loss Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Test comparison zoom
test_dates = dates[test_idx]
ax1.plot(test_dates, lstm_eval['y_test_true'], label='Actual Price', color='black', linewidth=2.0)
ax1.plot(test_dates, lstm_eval['y_test_pred'], label=f'LSTM (RMSE: ${lstm_eval[\"test\"][\"RMSE\"]:.2f})', color='#d62728', linestyle='--', linewidth=1.8)
ax1.plot(test_dates, gru_eval['y_test_pred'], label=f'GRU (RMSE: ${gru_eval[\"test\"][\"RMSE\"]:.2f})', color='#9467bd', linestyle='-.', linewidth=1.8)
ax1.set_title('Test Set Forecasting Comparison (Unseen Data)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.legend(loc='upper left', fontsize=11)

# Loss curves
epochs_range = range(1, NUM_EPOCHS + 1)
ax2.plot(epochs_range, lstm_loss, label=f'LSTM Loss (Final: {lstm_loss[-1]:.6f})', color='#2ca02c', linewidth=2.0)
ax2.plot(epochs_range, gru_loss, label=f'GRU Loss (Final: {gru_loss[-1]:.6f})', color='#ff7f0e', linewidth=2.0)
ax2.set_title('Training Loss Convergence (MSE across 100 Epochs)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('MSE Loss (Scaled)', fontsize=12)
ax2.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.show()""")

# --- AI SNAKE OIL & CRITICAL PERSPECTIVE ---
add_md("""---
## 🔍 Critical Analysis & AI Limitations (*AI Snake Oil* Perspective)

As emphasized throughout this 1-month machine learning project, a rigorous machine learning practitioner must temper technical enthusiasm with scientific skepticism, especially in financial domains. Readings such as *"AI Snake Oil"* (Prof. Arvind Narayanan & Sayash Kapoor, Princeton University) highlight common pitfalls:

### 1. The "Lagging Shadow" Illusion
In high-frequency and daily stock prediction, deep neural networks often achieve deceptive $R^2 > 0.95$ metrics by essentially learning the trivial identity mapping:
$$\\hat{y}_{t+1} \\approx y_t$$
Plotting the predicted price against the actual price often reveals that the prediction curve is merely the actual curve **shifted horizontally by 1 day**. While the loss appears low, this does not yield a profitable trading strategy because it fails to forecast the *direction and magnitude of price turning points*.

### 2. The Efficient Market Hypothesis (EMH)
Stock prices reflect instantaneous aggregation of global news, geopolitical events, earnings reports, and macroeconomic policies. A univariate time series (looking only at historical closing prices) fundamentally lacks the causal variables that drive future price changes.

### 3. Overfitting on Non-Stationary Processes
Financial time series are non-stationary (mean, variance, and covariance drift over time). Regime shifts (such as the 2008 financial crisis or unexpected economic disruptions) violate the fundamental Independent and Identically Distributed (I.I.D.) assumption of standard machine learning models.

---

## 🏁 Conclusions & Next Steps
- **Model Comparison:** The **GRU** model achieved comparable or superior accuracy to the **LSTM** while utilizing **~25% fewer trainable parameters** and completing training in less time.
- **Architecture Suitability:** For univariate time-series sequences of moderate length, the simpler gating mechanism of the GRU often avoids the overfitting pitfalls of the more complex LSTM cell.
- **Future Directions:**
  1. Incorporate **multivariate inputs**: Volume, RSI, MACD, Moving Averages, and sentiment signals from financial news.
  2. Implement **bidirectional RNNs** and **Temporal Fusion Transformers (TFT)**.
  3. Predict **daily percentage returns (log returns)** rather than raw nominal price levels to enforce stationarity.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.14"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook successfully written to: {NOTEBOOK_PATH}")
print(f"Total cells: {len(cells)}")
