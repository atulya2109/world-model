import torch
import torch.nn as nn
import numpy as np

LATENT_DIM = 256
ACTION_DIM = 3
SEQ_LEN = 32


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(32768, LATENT_DIM)  # 512x8x8
        self.fc_logvar = nn.Linear(32768, LATENT_DIM)

        # Decoder
        self.decoder_input = nn.Linear(LATENT_DIM, 32768)  # 512 * 8 * 8
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)

        z_projected = self.decoder_input(z)
        z_reshaped = z_projected.view(-1, 512, 8, 8)
        reconstruction = self.decoder(z_reshaped)

        return reconstruction, mu, logvar


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.d_model = 256

        self.z_emb = nn.Linear(LATENT_DIM, self.d_model)
        self.a_emb = nn.Linear(ACTION_DIM, self.d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.predict_head = nn.Linear(self.d_model, LATENT_DIM)

    def forward(self, z, action):
        # z: (Batch, Seq, 128)
        # action: (Batch, Seq, 3)

        # Fuse Vision + Action
        x = self.z_emb(z) + self.a_emb(action)

        # Add Position
        x += self.pos_emb[:, : x.size(1), :]

        # Causal Mask (Ensure we can't see the future while training)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Think
        out = self.transformer(x, mask=mask, is_causal=True)

        # Predict
        return self.predict_head(out)
