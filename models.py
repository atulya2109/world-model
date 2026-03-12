import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25) -> None:
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z_e: torch.Tensor):
        z_e_perm = z_e.permute(0, 2, 3, 1)

        distances= torch.cdist(z_e_perm.reshape(-1, z_e_perm.size(-1)), self.codebook.weight)

        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices).view(z_e_perm.shape)

        codebook_loss = F.mse_loss(z_q, z_e_perm.detach())
        commit_loss = F.mse_loss(z_q.detach(), z_e_perm)

        z_q_st = z_e_perm + (z_q - z_e_perm).detach()

        z_q_st = z_q_st.permute(0, 3, 1, 2)

        loss = codebook_loss + self.commitment_cost * commit_loss

        return z_q_st, loss, indices
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)
    
CITYSCAPES_CLASSES = 19


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=128, num_seg_classes=CITYSCAPES_CLASSES):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            ResBlock(128),
            ResBlock(128),
            nn.Conv2d(128, embedding_dim, 1),  # project to codebook dim
        )

        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, 1),  # project back up
            ResBlock(128),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        # Segmentation head: takes z_q (1/8 res) → full-res class logits
        self.seg_head = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_seg_classes, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        seg_logits = self.seg_head(z_q)
        # Ensure exact spatial match with input in case of non-divisible sizes
        seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x_recon, vq_loss, indices, seg_logits


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
