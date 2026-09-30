"""
PyTorch Neural Network Architectures: LSTM and GRU for Stock Price Regression.
Includes model parameter counts and flexible layer/dropout configuration.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) Recurrent Neural Network for Time Series Forecasting.
    
    Architecture:
        - Multi-layer LSTM with hidden states h_t and cell states c_t
        - Linear Readout Head mapping final hidden representation to predicted scalar price.
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.0
    ):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM layer: input shape (batch_size, seq_len, input_dim)
        dropout_rate = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Fully connected readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden state and cell state with zeros on the same device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        
        # Forward propagate through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    """
    Gated Recurrent Unit (GRU) Neural Network for Time Series Forecasting.
    
    Architecture:
        - Multi-layer GRU with reset and update gates (no separate cell state)
        - Linear Readout Head mapping final hidden representation to predicted scalar price.
        - Typically faster to train than LSTM due to fewer gate parameters.
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.0
    ):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # GRU layer: input shape (batch_size, seq_len, input_dim)
        dropout_rate = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Fully connected readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden state with zeros on the same device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        
        # Forward propagate through GRU
        out, hn = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
