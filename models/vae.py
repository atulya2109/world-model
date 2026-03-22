import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

LATENT_DIM = 128
ACTION_DIM = 3
SEQ_LEN = 32

    
class VectorQuantizer(nn.Module):
    ema_cluster_size: torch.Tensor
    ema_dw: torch.Tensor

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.5, decay: float = 0.99) -> None:
        super().__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

        self.register_buffer("ema_cluster_size", torch.ones(num_embeddings))
        self.register_buffer("ema_dw", self.codebook.weight.data.clone())

    def forward(self, z_e: torch.Tensor):
        z_e_perm = z_e.permute(0, 2, 3, 1)                          # (B, H, W, D)
        flat = z_e_perm.reshape(-1, z_e_perm.size(-1))               # (N, D)

        distances = torch.cdist(flat, self.codebook.weight)
        indices = distances.argmin(dim=-1)                            # (N,)
        z_q = self.codebook(indices).view(z_e_perm.shape)

        if self.training:
            # Cast to float32 — flat may be float16 under autocast
            flat32 = flat.float()

            # EMA update using scatter instead of dense one-hot to save memory
            counts = torch.bincount(indices, minlength=self.codebook.num_embeddings).float()
            dw = torch.zeros(self.codebook.num_embeddings, flat32.size(1), device=z_e.device)
            dw.scatter_add_(0, indices.unsqueeze(1).expand_as(flat32), flat32)

            self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
            self.ema_dw.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Laplace smoothing to avoid division by zero, then update codebook
            n = self.ema_cluster_size.sum()
            smoothed = (self.ema_cluster_size + 1e-5) / (n + self.codebook.num_embeddings * 1e-5) * n
            self.codebook.weight.data.copy_(self.ema_dw / smoothed.unsqueeze(1))

            # Dead code restart: reset unused codes to random encoder outputs
            dead = self.ema_cluster_size < 1.0
            if dead.any():
                random_flat = flat32[torch.randint(flat32.size(0), (int(dead.sum().item()),), device=z_e.device)]
                self.codebook.weight.data[dead] = random_flat
                self.ema_dw[dead] = random_flat
                self.ema_cluster_size[dead] = 1.0

        # Only commitment loss — codebook is updated via EMA, not gradients
        commit_loss = F.mse_loss(z_q.detach(), z_e_perm)
        z_q_st = z_e_perm + (z_q - z_e_perm).detach()
        z_q_st = z_q_st.permute(0, 3, 1, 2)

        return z_q_st, self.commitment_cost * commit_loss, indices
    
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
        return x_recon, vq_loss, indices, seg_logits, z_e


