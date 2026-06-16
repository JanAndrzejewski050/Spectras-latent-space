import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_out = F.softplus(self.decode(z))
        return x_out, mu, logvar
    
def vae_loss(preds, targets, mu, logvar, beta=0.1):
    rec = F.l1_loss(preds, targets)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kl, rec, kl


# Best Model
class ConvVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8,  kernel_size=17, stride=4, padding=8),
            nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=17, stride=4, padding=8),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=17, stride=4, padding=8),
            nn.GELU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            enc_out = self.encoder(dummy)
            self.enc_shape = enc_out.shape[1:]  # (32, L')
            flat_dim = enc_out.numel()
        print(flat_dim)

        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=17, stride=4, padding=8, output_padding=3),
            nn.GELU(),
            nn.ConvTranspose1d(16,  8, kernel_size=17, stride=4, padding=8, output_padding=3),
            nn.GELU(),
            nn.ConvTranspose1d( 8,  1, kernel_size=17, stride=4, padding=8, output_padding=3),
        )
        self.output_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3)
        self.input_dim = input_dim

    def encode(self, x):
        x = x.unsqueeze(1)
        h = self.encoder(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return mu

    def decode(self, z):
        h = F.gelu(self.fc_decode(z))                    # (B, flat_dim)
        h = h.view(z.size(0), *self.enc_shape)           # (B, 32, L')
        h = self.decoder(h)                              # (B, 1, 10048)
        h = h[..., :self.input_dim]
        h = self.output_conv(h)                          
        return h.squeeze(1)                              # (B, 10000)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_out      = F.softplus(self.decode(z))
        return x_out, mu, logvar
    
def vae_loss(preds, targets, mu, logvar, beta=0.1):
    rec = F.mse_loss(preds, targets)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kl, rec, kl



class ResBlock1d(nn.Module):
    """Residual block at fixed resolution (no stride). Used between strided layers."""
    def __init__(self, channels, kernel_size=17, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels),  # depthwise
            nn.Conv1d(channels, channels, kernel_size=1),                                                      # pointwise
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    def forward(self, x):
        return F.gelu(self.norm(x + self.block(x)))


class DownBlock(nn.Module):
    """Strided conv to halve/quarter resolution + one residual refinement."""
    def __init__(self, in_ch, out_ch, stride=4, dropout=0.1):
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=17, stride=stride, padding=8)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.res  = ResBlock1d(out_ch, dropout=dropout)

    def forward(self, x):
        return self.res(F.gelu(self.norm(self.down(x))))


class UpBlock(nn.Module):
    """Strided convtranspose to double/quadruple resolution + one residual refinement."""
    def __init__(self, in_ch, out_ch, stride=4, dropout=0.1):
        super().__init__()
        self.up   = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=17, stride=stride,
                                        padding=8, output_padding=stride-1)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.res  = ResBlock1d(out_ch, dropout=dropout)

    def forward(self, x):
        return self.res(F.gelu(self.norm(self.up(x))))


class ConvVAERes(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.1):
        super().__init__()

        # --- Encoder ---
        self.enc_in = nn.Conv1d(1, 8, kernel_size=1)   # channel lift, no spatial change
        self.down1  = DownBlock( 8, 16, stride=4, dropout=dropout)
        self.down2  = DownBlock(16, 32, stride=4, dropout=dropout)
        self.down3  = DownBlock(32, 32, stride=4, dropout=dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            enc_out = self._encode_conv(dummy)
            self.enc_shape = enc_out.shape[1:]   # (32, ~157)
            flat_dim = enc_out.numel()
            print(f"flat_dim: {flat_dim}")

        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        # --- Decoder (mirrors encoder) ---
        self.up1     = UpBlock(32, 32, stride=4, dropout=dropout)
        self.up2     = UpBlock(32, 16, stride=4, dropout=dropout)
        self.up3     = UpBlock(16,  8, stride=4, dropout=dropout)
        self.dec_out = nn.Conv1d(8, 1, kernel_size=7, padding=3)  # channel collapse + sharpening

        self.input_dim = input_dim

    def _encode_conv(self, x):
        x = F.gelu(self.enc_in(x))
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x

    def encode(self, x):
        h = self._encode_conv(x.unsqueeze(1)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return mu

    def decode(self, z):
        h = F.gelu(self.fc_decode(z))
        h = h.view(z.size(0), *self.enc_shape)   # (B, 32, ~157)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = h[..., :self.input_dim]              # trim to exact length
        return self.dec_out(h).squeeze(1)        # (B, 10000)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return F.softplus(self.decode(z)), mu, logvar


def vae_loss(preds, targets, mu, logvar, beta=0.1):
    rec = F.mse_loss(preds, targets)
    kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kl, rec, kl