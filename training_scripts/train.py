import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import pathlib
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gain_clamp", type=float, default=6.0)

    p.add_argument("--tweedie_p", type=float, default=1.3)
    p.add_argument("--peak_boost", type=float, default=8.0)
    p.add_argument("--under_penalty", type=float, default=2.0)
    p.add_argument("--peak_kernel", type=int, default=5)
    p.add_argument("--kl_warmup_steps", type=int, default=5000)
    p.add_argument("--kl_max_weight", type=float, default=1e-3)
    p.add_argument("--kl_cycles", type=int, default=4)
    p.add_argument("--free_bits", type=float, default=0.05)
    p.add_argument("--loss_balancing", type=str, default="learned",
                    choices=["learned", "fixed"])
    p.add_argument("--amp_weight", type=float, default=0.5)
    p.add_argument("--peak_match_weight", type=float, default=0.3)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")

    p.add_argument("--wandb_project", type=str, default="spectra-vae")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--sweep", action="store_true")
    return p.parse_args()


class SEBlock1d(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = x.mean(dim=-1)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class ConvVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, base=32, dropout=0.1, gain_clamp=6.0):
        super().__init__()
        self.gain_clamp = gain_clamp
        self.beta = nn.Parameter(torch.full((3,), 8.0))

        def enc_block(cin, cout):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=17, stride=4, padding=8),
                nn.GroupNorm(min(8, cout), cout),
                nn.GELU(),
            )

        self.enc1, self.se1 = enc_block(3, base // 2), SEBlock1d(base // 2)
        self.enc2, self.se2 = enc_block(base // 2, base), SEBlock1d(base)
        self.enc3, self.se3 = enc_block(base, base), SEBlock1d(base)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_dim)
            h = self.se3(self.enc3(self.se2(self.enc2(self.se1(self.enc1(dummy))))))
            self.enc_shape = h.shape[1:]
            flat_dim = h.numel()

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        self.gain_head = nn.Sequential(nn.Linear(latent_dim, 32), nn.GELU(), nn.Linear(32, 3))

        def dec_block(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose1d(cin, cout, kernel_size=17, stride=4, padding=8, output_padding=3),
                nn.GroupNorm(min(8, cout), cout),
                nn.GELU(),
            )

        self.dec1, self.sd1 = dec_block(base, base), SEBlock1d(base)
        self.dec2, self.sd2 = dec_block(base, base // 2), SEBlock1d(base // 2)
        self.dec3 = nn.ConvTranspose1d(base // 2, 3, kernel_size=17, stride=4, padding=8, output_padding=3)

        self.output_conv = nn.Conv1d(3, 3, kernel_size=7, padding=3)
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim

    def encode(self, x):
        h = self.se1(self.enc1(x))
        h = self.se2(self.enc2(h))
        h = self.se3(self.enc3(h))
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return mu

    def decode(self, z):
        log_gain = self.gain_head(z).clamp(-self.gain_clamp, self.gain_clamp)
        h = self.dropout(F.gelu(self.fc_decode(z)))
        h = h.view(z.size(0), *self.enc_shape)
        h = self.sd1(self.dec1(h))
        h = self.sd2(self.dec2(h))
        h = self.dec3(h)[..., :self.input_dim]
        shape = self.output_conv(h)
        beta = self.beta.view(1, 3, 1).clamp_min(1e-3)
        shape = F.softplus(shape * beta) / beta
        gain = log_gain.exp().unsqueeze(-1)
        return shape * gain

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_out = self.decode(z)
        return x_out, mu, logvar


class LossBalancer(nn.Module):
    def __init__(self, term_names):
        super().__init__()
        self.term_names = term_names
        self.log_sigma = nn.Parameter(torch.zeros(len(term_names)))

    def forward(self, losses):
        total = 0.0
        weights = {}
        for i, name in enumerate(self.term_names):
            precision = torch.exp(-self.log_sigma[i])
            total = total + precision * losses[name] + self.log_sigma[i]
            weights[name] = precision.item()
        return total, weights


def tweedie_deviance(pred, target, p, eps=1e-6):
    pred = pred.clamp_min(eps)
    return -target * pred.pow(1 - p) / (1 - p) + pred.pow(2 - p) / (2 - p)


def peak_height_loss(pred, target, kernel=5):
    pooled = F.max_pool1d(target, kernel, stride=1, padding=kernel // 2)
    is_peak = (target == pooled) & (target > 0)
    if is_peak.sum() == 0:
        return torch.zeros((), device=pred.device)
    return F.mse_loss(pred[is_peak], target[is_peak])


def cyclical_kl_weight(step, total_steps, cycles, max_weight, ratio=0.5):
    if total_steps <= 0 or cycles <= 0:
        return max_weight
    period = total_steps / cycles
    pos = (step % period) / period
    return max_weight * min(1.0, pos / ratio)


def free_bits_kl(mu, logvar, free_bits):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.mean()


def compute_loss(pred, target, mu, logvar, step, total_steps, args, balancer):
    eps = 1e-6
    peak_mask = (target > eps).float()

    dev = tweedie_deviance(pred, target, p=args.tweedie_p, eps=eps)
    tweedie_loss = (dev * (1.0 + args.peak_boost * peak_mask)).mean()

    diff = target - pred
    asym = torch.where(diff > 0, args.under_penalty * diff ** 2, diff ** 2)
    amp_loss = (asym * peak_mask).sum() / peak_mask.sum().clamp_min(1.0)

    peak_match = peak_height_loss(pred, target, kernel=args.peak_kernel)

    kl = free_bits_kl(mu, logvar, args.free_bits)
    kl_w = cyclical_kl_weight(step, total_steps, args.kl_cycles, args.kl_max_weight)

    if args.loss_balancing == "learned":
        losses = {"tweedie": tweedie_loss, "amp": amp_loss, "peak_match": peak_match, "kl": kl_w * kl}
        total, weights = balancer(losses)
    else:
        total = tweedie_loss + args.amp_weight * amp_loss + args.peak_match_weight * peak_match + kl_w * kl
        weights = {}

    logs = {
        "tweedie": tweedie_loss.item(),
        "amp": amp_loss.item(),
        "peak_match": peak_match.item(),
        "kl": kl.item(),
        "kl_weight": kl_w,
    }
    logs.update({f"w_{k}": v for k, v in weights.items()})
    return total, logs



def load_data(args):
    data_dir = Path('/home/andrze06/projects/Spectras-latent-space/data/multimodal_spectroscopic_dataset')
    files = sorted(data_dir.glob("aligned_chunk_*.parquet"))
    dfs = [pd.read_parquet(f, columns=['smiles', 'c_nmr_spectra', 'h_nmr_spectra', 'msms_cfmid_positive_20ev']) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    MS_BINS = 10000
    MAX_MZ = 1100.0        # typical MS range
    LOG_SCALE = False      # optional
    THRESHOLD = 1e-4

    def ms_to_dense(ms_peaks, bins=10000, max_mz=1000):

        spectrum = np.zeros(bins, dtype=np.float32)

        if ms_peaks is None or len(ms_peaks) == 0:
            return spectrum

        peaks = np.array([np.asarray(p, dtype=np.float32) for p in ms_peaks])

        mz = peaks[:, 0]
        intensity = peaks[:, 1]

        idx = np.clip(
            (mz / max_mz * (bins - 1)).astype(np.int32),
            0,
            bins - 1,
        )

        np.maximum.at(spectrum, idx, intensity)

        return spectrum

    df["ms_spectra"] = [
        ms_to_dense(x)
        for x in tqdm(df["msms_cfmid_positive_20ev"])
    ]

    ms_spectra = df['ms_spectra'].to_list()
    c_nmr_data = df['c_nmr_spectra'].to_list()
    h_nmr_data = df['h_nmr_spectra'].to_list()

    c_max = max(np.max(x) for x in c_nmr_data)
    h_max = max(np.max(x) for x in h_nmr_data)
    ms_max = max(np.max(x) for x in ms_spectra)

    ms = gaussian_filter1d(ms_spectra, sigma=2.5)

    def peak_stats(spectra, threshold=THRESHOLD):
        s = 0.0
        s2 = 0.0
        n = 0

        for spec in tqdm(spectra):
            peaks = spec[spec > threshold]

            if len(peaks) == 0:
                continue

            s += peaks.sum(dtype=np.float64)
            s2 += np.square(peaks, dtype=np.float64).sum(dtype=np.float64)
            n += len(peaks)

        mean = s / n
        std = np.sqrt(s2 / n - mean**2)

        return mean, std

    c_mean, c_std = peak_stats(c_nmr_data)
    h_mean, h_std = peak_stats(h_nmr_data)
    ms_mean, ms_std = peak_stats(ms)

    class SpectraDataset(Dataset):
        def __init__(self, c_nmr, h_nmr, ms, c_max, h_max, ms_max, indices, scale=1e4):
            self.c_nmr = c_nmr
            self.h_nmr = h_nmr
            self.ms = ms
            self.c_max = c_max
            self.h_max = h_max
            self.ms_max = ms_max
            self.indices = indices
            self.scale = scale

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i = self.indices[idx]

            # c = normalize_peaks(self.c_nmr[idx], c_mean, c_std)
            # h = normalize_peaks(self.h_nmr[idx], h_mean, h_std)
            # ms = normalize_peaks(self.ms[idx], ms_mean, ms_std)
            c = self.c_nmr[i] / (c_std)#np.log1p((self.c_nmr[i] / self.c_max) * self.scale) #
            h = self.h_nmr[i] / (h_std)#np.log1p((self.h_nmr[i] / self.h_max) * self.scale) #
            ms = self.ms[i] / (ms_std)#np.log1p((self.ms[i] / self.ms_max) * self.scale) #

            x = np.stack((c, h, ms), axis=0).astype(np.float32)
            return torch.from_numpy(x)  

    idx = np.arange(len(c_nmr_data))
    train_idx, dummy_idx = train_test_split(idx, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(dummy_idx, test_size=0.5, random_state=42)

    train_data = SpectraDataset(c_nmr_data, h_nmr_data, ms, c_max, h_max, ms_max, train_idx)
    val_data = SpectraDataset(c_nmr_data, h_nmr_data, ms, c_max, h_max, ms_max, val_idx)
    test_data = SpectraDataset(c_nmr_data, h_nmr_data, ms, c_max, h_max, ms_max, test_idx)

    return train_data, val_data, train_data[0].shape[0]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                      name=args.run_name, config=vars(args))
    if args.sweep:
        for k, v in wandb.config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_data, val_data, input_dim = load_data(args)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, pin_memory=True)

    model = ConvVAE(input_dim, args.latent_dim, base=args.base_channels,
                     dropout=args.dropout, gain_clamp=args.gain_clamp).to(device)

    balancer = LossBalancer(["tweedie", "amp", "peak_match", "kl"]).to(device)
    params = list(model.parameters())
    if args.loss_balancing == "learned":
        params += list(balancer.parameters())

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    wandb.watch(model, log="gradients", log_freq=200)

    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        logs_accum = {}

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for x in pbar:
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                x_pred, mu, logvar = model(x)
                loss, logs = compute_loss(x_pred, x, mu, logvar, global_step, total_steps, args, balancer)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix({k: f"{v:.4f}" for k, v in list(logs.items())[:4]})
            train_loss += loss.item()
            for k, v in logs.items():
                logs_accum[k] = logs_accum.get(k, 0) + v
            global_step += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    x_pred, mu, logvar = model(x)
                    loss, _ = compute_loss(x_pred, x, mu, logvar, global_step, total_steps, args, balancer)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        logs_accum = {k: v / len(train_loader) for k, v in logs_accum.items()}

        wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                    "lr": scheduler.get_last_lr()[0], "epoch": epoch, **logs_accum}, step=global_step)
        print(f"epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       os.path.join(args.ckpt_dir, f"{run.id}_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    wandb.finish()


if __name__ == "__main__":
    main()
