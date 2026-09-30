"""
Complete End-to-End Orchestrator:
Trains LSTM & GRU, Evaluates Performance, Saves Figures & Results, and Runs Forecasting Demo.
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torch

from src.train import run_training_experiment
from src.evaluate import evaluate_model, plot_predictions_and_loss
from src.predict import predict_next_day


def main():
    print("=" * 70)
    print("[BENCHMARK] STOCK PRICE PREDICTION: PYTORCH LSTM vs GRU")
    print("=" * 70)
    
    data_file = os.path.join(PROJECT_ROOT, "data", "AMZN.csv")
    if not os.path.exists(data_file):
        print("Data file not found. Running download_data script...")
        from scripts.download_data import download_dataset
        download_dataset()

    # 1. Run training experiment
    results_dir = os.path.join(PROJECT_ROOT, "results")
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    lookback = 20
    epochs = 100
    lr = 0.01
    hidden_dim = 32
    num_layers = 2
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Lookback: {lookback} | Epochs: {epochs} | LR: {lr}\n")
    
    experiment = run_training_experiment(
        data_path=data_file,
        lookback=lookback,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_epochs=epochs,
        learning_rate=lr,
        save_dir=results_dir,
        device=device
    )
    
    scaler = experiment["scaler"]
    raw_df = experiment["raw_df"]
    X_train_t, y_train_t, X_test_t, y_test_t = experiment["tensors"]
    
    # 2. Evaluate both models
    print("\n" + "=" * 70)
    print("EVALUATING MODELS ON UNSEEN TEST DATA (20% Split)")
    print("=" * 70)
    
    lstm_eval = evaluate_model(experiment["lstm"]["model"], X_train_t, y_train_t, X_test_t, y_test_t, scaler, device=device)
    gru_eval = evaluate_model(experiment["gru"]["model"], X_train_t, y_train_t, X_test_t, y_test_t, scaler, device=device)
    
    # 3. Create comparison DataFrame
    from src.models import count_parameters
    lstm_params = count_parameters(experiment["lstm"]["model"])
    gru_params = count_parameters(experiment["gru"]["model"])
    
    comp_rows = [
        {
            "Model": "LSTM",
            "Parameters": lstm_params,
            "Training Time (s)": round(experiment["lstm"]["time"], 2),
            "Train RMSE ($)": lstm_eval["train_metrics"]["RMSE"],
            "Test RMSE ($)": lstm_eval["test_metrics"]["RMSE"],
            "Test MSE": lstm_eval["test_metrics"]["MSE"],
            "Test MAE ($)": lstm_eval["test_metrics"]["MAE"],
            "Test MAPE (%)": lstm_eval["test_metrics"]["MAPE(%)"],
            "Test R2": lstm_eval["test_metrics"]["R2"]
        },
        {
            "Model": "GRU",
            "Parameters": gru_params,
            "Training Time (s)": round(experiment["gru"]["time"], 2),
            "Train RMSE ($)": gru_eval["train_metrics"]["RMSE"],
            "Test RMSE ($)": gru_eval["test_metrics"]["RMSE"],
            "Test MSE": gru_eval["test_metrics"]["MSE"],
            "Test MAE ($)": gru_eval["test_metrics"]["MAE"],
            "Test MAPE (%)": gru_eval["test_metrics"]["MAPE(%)"],
            "Test R2": gru_eval["test_metrics"]["R2"]
        }
    ]
    
    comp_df = pd.DataFrame(comp_rows)
    print("\n" + comp_df.to_string(index=False))
    
    # Save comparison table
    comp_csv_path = os.path.join(results_dir, "model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved comparison metrics table to: {comp_csv_path}")
    
    # 4. Generate plots
    print("\nGenerating evaluation plots...")
    plot_predictions_and_loss(
        raw_df=raw_df,
        lookback=lookback,
        lstm_results=lstm_eval,
        gru_results=gru_eval,
        lstm_loss=experiment["lstm"]["loss"],
        gru_loss=experiment["gru"]["loss"],
        save_dir=figures_dir
    )
    
    # 5. Run prediction demo
    print("\n" + "=" * 70)
    print("NEXT-DAY STOCK PRICE FORECASTING DEMO")
    print("=" * 70)
    predict_next_day(
        model_type="gru",
        weights_path=os.path.join(results_dir, "gru_model.pth"),
        data_path=data_file,
        lookback=lookback,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device
    )
    
    print("\n" + "=" * 70)
    print("[SUCCESS] FULL BENCHMARK EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
