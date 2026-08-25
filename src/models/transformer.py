
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class UnsupervisedTemporalTransformer(nn.Module):
    """
    UPGRADE 1: Lightweight unsupervised Transformer Encoder.
    Captures temporal dependencies in latent quantum/physics features.
    """
    def __init__(self, input_dim=8, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, input_dim) # For reconstruction loss

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        x = self.embedding(x)
        x = self.pos_encoder(x)
        features = self.transformer_encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction, features

def get_transformer_reconstruction_error(model, x):
    """Computes the anomaly score as standard reconstruction error."""
    model.eval()
    with torch.no_grad():
        recon, _ = model(x)
        error = torch.mean((x - recon)**2, dim=(1,2))
    return error.numpy()
