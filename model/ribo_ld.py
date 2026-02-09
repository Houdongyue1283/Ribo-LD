import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, seq_len, latent_dim=64):
        super().__init__()
        self.seq_len = seq_len

        self.conv_layers = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=5, padding=2),
            nn.LayerNorm([16, seq_len]),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.LayerNorm([32, seq_len // 2]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Simple transformer encoder - single layer, fewer heads
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=2, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)

        self.conv_output_size = 32 * (seq_len // 4)
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, self.seq_len, 5).permute(0, 2, 1)

        x = self.conv_layers(x_reshaped)  # [B, 32, L]
        x = x.transpose(1, 2)  # [B, L, 32]
        x = self.transformer_encoder(x)  # [B, L, 32]
        x = x.transpose(1, 2).contiguous()  # [B, 32, L]

        encoded_flat = x.view(batch_size, -1)
        z = self.fc(encoded_flat)

        return z


class Decoder(nn.Module):
    def __init__(self, seq_len, latent_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.conv_output_size = 32 * (seq_len // 4)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.conv_output_size),
            nn.LayerNorm(self.conv_output_size),
            nn.ReLU()
        )

        # add dropout to decoder fc
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.conv_output_size),
            nn.LayerNorm(self.conv_output_size),
            nn.ReLU(),

        )

        # Simple transformer decoder - single layer, fewer heads  
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=32, nhead=2, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=1)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LayerNorm([16, seq_len // 2]),
            nn.ReLU(),


            nn.ConvTranspose1d(16, 5, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LayerNorm([5, seq_len]),
            nn.ReLU(),

        )

    def forward(self, z):
        batch_size = z.size(0)

        decoded_flat = self.fc(z)
        decoded_reshaped = decoded_flat.view(batch_size, 32, self.seq_len // 4)

        x = decoded_reshaped.transpose(1, 2)  # [B, L, 32]
        x = self.transformer_decoder(x, x)  # [B, L, 32]
        x = x.transpose(1, 2).contiguous()  # [B, 32, L]

        x_rec = self.deconv_layers(x)
        x_rec = x_rec.permute(0, 2, 1).contiguous().view(batch_size, -1)

        return x_rec



class Ribo_LD_AEWithRegressor(nn.Module):
    def __init__(self, seq_len, latent_dim=64):
        super().__init__()
        
        self.encoder = Encoder(seq_len, latent_dim)
        self.decoder = Decoder(seq_len, latent_dim)
        
        self.activity_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),        # Prevent overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    
    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.activity_predictor(z)
        x_rec = self.decoder(z)
        return x_rec, y_pred
