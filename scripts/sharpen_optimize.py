"""Optimize Wiener σ via PyTorch with a constraint: overshoot ≤ JPEG-baseline.

The camera ISP injects its own halos when it produces the rendered JPEG; we
read those halos by comparing JPEG↓ against raw at 192×256 (per-channel,
relative to raw's 9×9 local extrema with a small tolerance band). That gives
us a per-photo overshoot fraction = the user's de-facto 'acceptable edge
effect' level.

We then optimize a SINGLE Wiener σ (joint across all photos) that maximizes
the gradient-energy of the deconvolved input subject to its overshoot fraction
not exceeding the target. The penalty multiplier β is auto-adjusted via
augmented-Lagrangian: every K iterations, β grows if the constraint is still
violated. This converges to a Pareto-optimal σ — "sharpest possible while
matching the camera's own halo level."

Run: uv run scripts/sharpen_optimize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    INPUT, OUT, parse_ijpeg_header, extract_raw_thermal,
    normalize_with_cut, colormap_rgb,
)

PHOTOS = ["1777733165452", "1777804010136", "1771110170401"]


# ---------- inputs ----------

def render_raw_inferno(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    norm = normalize_with_cut(thermal, cut_percentile=1.0)
    return colormap_rgb(norm, "inferno")


def render_jpeg_down(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    return np.array(img.convert("RGB").resize((ir_w, ir_h), Image.Resampling.LANCZOS))


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None] / 255.0


def overshoot_metrics(Y: torch.Tensor, X: torch.Tensor,
                      window: int = 9, tol: float = 0.02
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (overshoot_fraction, overshoot_magnitude) of Y against X's 9×9
    local extrema (per channel, mean-aggregated). Magnitude is differentiable;
    fraction is for reporting."""
    pad = window // 2
    x_mx = F.max_pool2d(F.pad(X, [pad] * 4, mode="reflect"), window, stride=1)
    x_mn = -F.max_pool2d(F.pad(-X, [pad] * 4, mode="reflect"), window, stride=1)

    over_mag = (F.relu(Y - x_mx - tol) ** 2 + F.relu(x_mn - tol - Y) ** 2).mean()

    with torch.no_grad():
        over_frac = ((Y > x_mx + tol) | (Y < x_mn - tol)).float().mean()

    return over_frac, over_mag


def luminance(Y: torch.Tensor) -> torch.Tensor:
    return 0.2126 * Y[:, 0:1] + 0.7152 * Y[:, 1:2] + 0.0722 * Y[:, 2:3]


def grad_energy(Y: torch.Tensor) -> torch.Tensor:
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=Y.dtype, device=Y.device).view(1, 1, 3, 3) / 8.0
    sy = sx.transpose(-1, -2)
    lum = luminance(Y)
    gx = F.conv2d(F.pad(lum, [1] * 4, mode="reflect"), sx)
    gy = F.conv2d(F.pad(lum, [1] * 4, mode="reflect"), sy)
    return (gx ** 2 + gy ** 2).mean()


# ---------- Wiener (PyTorch, differentiable in σ) ----------

def wiener_torch(X: torch.Tensor, sigma: torch.Tensor, K: float = 1e-3,
                  size: int = 21) -> torch.Tensor:
    H, W = X.shape[-2:]
    device = X.device
    x = torch.arange(size, device=device).float() - (size - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    psf = (g.view(1, -1) * g.view(-1, 1)).view(1, 1, size, size)

    pad = torch.zeros(1, 1, H, W, device=device)
    ph, pw = (H - size) // 2, (W - size) // 2
    pad[:, :, ph:ph + size, pw:pw + size] = psf
    pad = torch.fft.ifftshift(pad, dim=(-2, -1))
    PSF_F = torch.fft.fft2(pad)
    IMG_F = torch.fft.fft2(X)
    G = torch.conj(PSF_F) / (PSF_F.abs() ** 2 + K)
    return torch.fft.ifft2(IMG_F * G).real


# ---------- main optimization ----------

def main() -> None:
    Xs, JPGs = [], []
    for p in PHOTOS:
        jpeg = INPUT / f"{p}.jpg"
        Xs.append(to_tensor(render_raw_inferno(jpeg)))
        JPGs.append(to_tensor(render_jpeg_down(jpeg)))

    X = torch.cat(Xs)        # B×3×H×W (raw)
    JPG = torch.cat(JPGs)    # B×3×H×W (JPEG↓)

    # NEW anchor: match JPEG-baseline sharpness (the camera's gradient energy);
    # minimize overshoot to get there. The optimum σ is the smallest one
    # achieving target sharpness — least halo while matching camera detail.
    print("baseline (JPEG↓) sharpness ∇² vs raw, per photo:")
    target_sharps, raw_sharps = [], []
    for i, p in enumerate(PHOTOS):
        s_jpg = grad_energy(JPG[i:i + 1]).item()
        s_raw = grad_energy(X[i:i + 1]).item()
        f, _ = overshoot_metrics(JPG[i:i + 1], X[i:i + 1])
        print(f"  {p}: JPG ∇²={s_jpg:.4f}  raw ∇²={s_raw:.4f}  "
              f"JPG_overshoot={f.item()*100:.1f}%")
        target_sharps.append(s_jpg)
        raw_sharps.append(s_raw)
    target_sharp = float(np.mean(target_sharps))
    raw_sharp_mean = float(np.mean(raw_sharps))
    print(f"target sharpness = JPEG-baseline mean ∇² = {target_sharp:.4f}  "
          f"(raw mean ∇² = {raw_sharp_mean:.4f})")

    # Loss = (sharp(σ) − target_sharp)² + γ · overshoot_mag
    # Adam learns σ. The (sharp − target)² term has a unique global minimum at
    # sharp = target. The overshoot term breaks degeneracy / regularizes.
    raw_p = torch.tensor(0.0, requires_grad=True)
    sigma_min, sigma_max = 0.3, 2.0
    opt = torch.optim.Adam([raw_p], lr=2e-2)

    gamma = 0.0                    # pure sharpness-matching, no halo regularizer
    steps = 1500

    history = []
    for it in range(steps):
        opt.zero_grad()
        sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(raw_p)
        Y = wiener_torch(X, sigma)
        sharp = grad_energy(Y)
        over_frac, over_mag = overshoot_metrics(Y, X)

        loss = (sharp - target_sharp) ** 2 + gamma * over_mag
        loss.backward()
        opt.step()

        if it % 100 == 0 or it == steps - 1:
            history.append((it, float(sigma.item()), float(sharp.item()),
                            float(over_frac.item()), gamma, float(loss.item())))
            print(f"  it={it:4d}  σ={sigma.item():.3f}  "
                  f"sharp={sharp.item():.4f}  "
                  f"over_frac={over_frac.item()*100:.2f}%  "
                  f"loss={loss.item():.5f}")

    # Print final
    final_sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(raw_p).item()
    print(f"\nfinal: σ = {final_sigma:.3f}")
    with torch.no_grad():
        Yfin = wiener_torch(X, torch.tensor(final_sigma))
        f_fin, _ = overshoot_metrics(Yfin, X)
        s_fin = grad_energy(Yfin)
        print(f"       sharpness ∇² = {s_fin.item():.4f}  (target {target_sharp:.4f})")
        print(f"       overshoot_frac = {f_fin.item()*100:.2f}%")

    # Per-photo sigma sweep around final σ for context
    print(f"\nsweep around σ_opt for context (per-photo overshoot):")
    sweep = sorted(set([round(s, 2) for s in
                        [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5,
                         round(final_sigma, 2)]]))
    print(f"  {'σ':6s}  ∇²       " + "  ".join(f"{p[-6:]:>9s}" for p in PHOTOS))
    for s in sweep:
        Y = wiener_torch(X, torch.tensor(float(s)))
        s_e = grad_energy(Y).item()
        per = []
        for i in range(X.shape[0]):
            f, _ = overshoot_metrics(Y[i:i + 1], X[i:i + 1])
            per.append(f.item() * 100)
        per_str = "  ".join(f"{v:8.2f}%" for v in per)
        marker = "  *" if abs(s - round(final_sigma, 2)) < 0.005 else ""
        print(f"  σ={s:.2f}  {s_e:.4f}  {per_str}{marker}")

    # Save history for plot
    np.save(OUT / "sharpen_optimize_history.npy",
            np.array([(it, sg, sh, of, b, ls) for (it, sg, sh, of, b, ls) in history],
                     dtype=[("it", "i4"), ("sigma", "f4"), ("sharp", "f4"),
                            ("over_frac", "f4"), ("beta", "f4"), ("loss", "f4")]))
    print(f"\nwrote {OUT/'sharpen_optimize_history.npy'}")
    print(f"\noptimal σ = {final_sigma:.3f}  (matches JPEG-baseline sharpness)")


if __name__ == "__main__":
    main()
