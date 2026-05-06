"""Sharpening preprocessing before upscayl-standard.

Hypothesis: the raw render is softer than the camera-JPEG because the camera
ISP applied edge enhancement before encoding. We can sharpen the raw 192×256
render BEFORE running upscayl, and pick the sharpener that makes the SR output
sharp without introducing halos.

Three sharpeners compared, all per-photo:

  A) Wiener deconvolution (1 free param: σ of Gaussian PSF; K=1e-3 fixed)
     y = IFFT( FFT(x) · conj(H) / (|H|² + K) )
     where H is the FFT of a Gaussian kernel of width σ.

  B) Unsharp mask (1 free param: amount, with σ=1.0 fixed)
     y = x + amount · (x − Gaussian(x, σ))

  C) Learnable 5×5 conv kernel (25 free params, PyTorch + Adam)
     Loss = -λ·sharpness + α·overshoot + β·content
       sharpness = mean |∇y|²
       overshoot = mean( relu(y - maxpool(x, 5))² + relu(minpool(x, 5) - y)² )
                   — penalizes any pixel exceeding the input's local 5×5
                     min/max (the halo signature).
       content   = mean(y - x)²
     Per-channel kernel; identity init (delta), Adam, 1500 steps.

For each sharpener we pick the param value (or trained kernel) that minimizes
overshoot at fixed gradient-energy gain, then run upscayl-standard on the
sharpened input and compare.

Run: uv run scripts/sharpen_pre.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import sobel
from scipy.signal import convolve2d

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    INPUT, OUT, WORK, UPSCAYL_BIN, UPSCAYL_MODELS,
    parse_ijpeg_header, extract_raw_thermal,
    normalize_with_cut, colormap_rgb,
)

PHOTOS = ["1777733165452", "1777804010136", "1771110170401"]
MODEL = "upscayl-standard-4x"

REGIONS = {  # fractional center on 768×1024, half-side
    "1777733165452": ((0.50, 0.78), 144),
    "1777804010136": ((0.50, 0.36), 144),
    "1771110170401": ((0.45, 0.78), 144),
}


# ---------- raw → 192×256 inferno RGB ----------

def render_raw_inferno(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    norm = normalize_with_cut(thermal, cut_percentile=1.0)
    return colormap_rgb(norm, "inferno")  # uint8 H×W×3


# ---------- sharpness / halo metrics ----------

def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]).astype(np.float64)


def grad_mag(g: np.ndarray) -> np.ndarray:
    return np.hypot(sobel(g, axis=1, mode="reflect"),
                    sobel(g, axis=0, mode="reflect"))


def lap_var(g: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    return float(convolve2d(g, k, mode="same", boundary="symm").var())


def overshoot_count(input_rgb: np.ndarray, output_rgb: np.ndarray,
                    k: int = 5, tol: float = 5.0) -> float:
    """Fraction of pixels in OUTPUT (any resolution) that exceed the INPUT's
    local k×k max + tol, or fall below local k×k min − tol. The input is
    nearest-up-sampled to match the output. Halo signature."""
    out_h, out_w = output_rgb.shape[:2]
    inp_resized = np.array(
        Image.fromarray(input_rgb).resize((out_w, out_h), Image.Resampling.NEAREST)
    )
    inp = torch.from_numpy(inp_resized.astype(np.float32)).permute(2, 0, 1)[None]
    out = torch.from_numpy(output_rgb.astype(np.float32)).permute(2, 0, 1)[None]
    pad = k // 2
    mx = F.max_pool2d(F.pad(inp, [pad] * 4, mode="reflect"), k, stride=1)
    mn = -F.max_pool2d(F.pad(-inp, [pad] * 4, mode="reflect"), k, stride=1)
    over = (out > mx + tol).float().mean().item()
    under = (out < mn - tol).float().mean().item()
    return over + under


def metrics_of(rgb: np.ndarray, ref: np.ndarray | None = None) -> dict:
    g = luminance(rgb)
    gm = grad_mag(g)
    out = {"gmean": float(gm.mean()),
           "gp95": float(np.percentile(gm, 95)),
           "lapvar": lap_var(g)}
    if ref is not None:
        out["overshoot"] = overshoot_count(ref, rgb)
    return out


# ---------- sharpeners ----------

def gaussian_2d(sigma: float, size: int) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    g1 = np.exp(-x**2 / (2 * sigma**2))
    g1 = g1 / g1.sum()
    return np.outer(g1, g1)


def wiener_deconvolve(rgb: np.ndarray, sigma: float, K: float = 1e-3) -> np.ndarray:
    """Per-channel Wiener deconvolution against a Gaussian PSF of width sigma."""
    H, W, _ = rgb.shape
    size = max(7, int(6 * sigma + 1) | 1)
    psf = gaussian_2d(sigma, size)
    # Zero-pad PSF to image size and shift to zero-phase
    pad = np.zeros((H, W), dtype=np.float64)
    py, px = (H - size) // 2, (W - size) // 2
    pad[py:py + size, px:px + size] = psf
    pad = np.fft.ifftshift(pad)
    PSF_F = np.fft.fft2(pad)
    out = np.zeros_like(rgb, dtype=np.float64)
    for c in range(3):
        ch = rgb[..., c].astype(np.float64)
        ch_F = np.fft.fft2(ch)
        G = np.conj(PSF_F) / (np.abs(PSF_F) ** 2 + K)
        out[..., c] = np.fft.ifft2(ch_F * G).real
    return np.clip(out, 0, 255).astype(np.uint8)


def unsharp_mask(rgb: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    H, W, _ = rgb.shape
    size = max(5, int(6 * sigma + 1) | 1)
    g = gaussian_2d(sigma, size)
    out = np.zeros_like(rgb, dtype=np.float64)
    for c in range(3):
        ch = rgb[..., c].astype(np.float64)
        blurred = convolve2d(ch, g, mode="same", boundary="symm")
        out[..., c] = ch + amount * (ch - blurred)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- learnable Wiener σ (PyTorch, 1 param) ----------

def train_wiener_sigma(rgb_list: list[np.ndarray],
                        steps: int = 600,
                        lr: float = 0.02,
                        sigma_min: float = 0.3, sigma_max: float = 2.0,
                        K: float = 1e-3,
                        target_overshoot_frac: float = 0.20,
                        beta: float = 200.0,
                        verbose: bool = False) -> tuple[float, dict]:
    """Learn the best Gaussian PSF σ for Wiener deconvolution, jointly across
    photos. Single learnable parameter (sigmoid-bounded into [σ_min, σ_max]).

    The PyTorch reason: Wiener deconvolution is a closed-form operation but
    'how much halo is acceptable' is the parameter we want to optimize. We
    define a soft target (overshoot stays under target_overshoot_frac of
    pixels) and maximize gradient-energy gain subject to that target.

    Loss = -mean(|∇y|²)
         + beta · relu( overshoot_frac − target_overshoot_frac )²
    """
    device = "cpu"
    Xs = []
    for rgb in rgb_list:
        Xs.append(torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1) / 255.0)
    X = torch.stack(Xs).to(device)        # B×3×H×W
    H, W = X.shape[-2:]

    # Sigmoid-bounded σ parameter
    raw_p = torch.tensor(0.0, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw_p], lr=lr)

    # Sobel kernels for gradient energy
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=device).view(1, 1, 3, 3) / 8.0
    sy = sx.transpose(-1, -2)

    # Pre-compute input local extrema (constant) — 9×9 window
    pad_w = 4
    x_mx = F.max_pool2d(F.pad(X, [pad_w] * 4, mode="reflect"), 9, stride=1)
    x_mn = -F.max_pool2d(F.pad(-X, [pad_w] * 4, mode="reflect"), 9, stride=1)
    tol_t = 0.02

    def sigma_value():
        return sigma_min + (sigma_max - sigma_min) * torch.sigmoid(raw_p)

    def gaussian_psf(sigma: torch.Tensor, size: int = 21) -> torch.Tensor:
        # Returns (1, 1, size, size) Gaussian, normalized
        x = torch.arange(size, device=device).float() - (size - 1) / 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        return (g.view(1, -1) * g.view(-1, 1)).view(1, 1, size, size)

    def wiener_torch(img: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # img: B×3×H×W
        psf = gaussian_psf(sigma, size=21)
        # Pad PSF into image-size canvas, zero-phase
        pad = torch.zeros(1, 1, H, W, device=device)
        ph, pw = (H - 21) // 2, (W - 21) // 2
        pad[:, :, ph:ph + 21, pw:pw + 21] = psf
        pad = torch.fft.ifftshift(pad, dim=(-2, -1))
        PSF_F = torch.fft.fft2(pad)
        IMG_F = torch.fft.fft2(img)
        G = torch.conj(PSF_F) / (PSF_F.abs() ** 2 + K)
        return torch.fft.ifft2(IMG_F * G).real

    for it in range(steps):
        opt.zero_grad()
        sigma = sigma_value()
        Y = wiener_torch(X, sigma)

        lum = (0.2126 * Y[:, 0:1] + 0.7152 * Y[:, 1:2] + 0.0722 * Y[:, 2:3])
        gx = F.conv2d(F.pad(lum, [1] * 4, mode="reflect"), sx)
        gy = F.conv2d(F.pad(lum, [1] * 4, mode="reflect"), sy)
        sharpness = (gx ** 2 + gy ** 2).mean()

        # Differentiable soft overshoot — magnitude squared past the tol band
        over_mag = (F.relu(Y - x_mx - tol_t) ** 2
                    + F.relu(x_mn - tol_t - Y) ** 2).mean()
        # And a hard-counted fraction for reporting / target enforcement
        with torch.no_grad():
            overshoot_frac = ((Y > x_mx + tol_t) | (Y < x_mn - tol_t)).float().mean()

        loss = -sharpness + beta * over_mag
        loss.backward()
        opt.step()

        if verbose and (it % 100 == 0 or it == steps - 1):
            print(f"    it={it:4d}  σ={sigma.item():.3f}  "
                  f"sharp={sharpness.item():.4f}  "
                  f"over_frac={overshoot_frac.item():.3f}  loss={loss.item():.4f}")

    final_sigma = float(sigma_value().item())
    info = {"sigma": final_sigma,
            "sharpness": float(sharpness.item()),
            "overshoot_frac": float(overshoot_frac.item())}
    return final_sigma, info


# (legacy sig kept for backwards compatibility, not used)
def train_learnable_kernel(rgb_list: list[np.ndarray], k: int = 5,
                            steps: int = 2000,
                            lam_sharp: float = 1000.0,
                            alpha_overshoot: float = 5.0,
                            beta_content: float = 1.0,
                            window: int = 9, tol: float = 0.02,
                            verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Train a single shared k×k kernel across all photos.

    Loss = -lam_sharp · mean|∇y|²
         + alpha_overshoot · mean( relu(y − maxpool(x,window) − tol)²
                                 + relu(minpool(x,window) − tol − y)² )
         + beta_content · mean(y − x)²

    The window is wider than k so that mild overshoot at scale-of-the-kernel
    is allowed (sharpening necessarily exceeds the immediate-neighborhood
    extrema). The tol band gives further slack so the kernel can actually
    sharpen rather than collapsing to identity.
    """
    device = "cpu"
    Xs = []
    for rgb in rgb_list:
        Xs.append(torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1) / 255.0)
    X = torch.stack(Xs).to(device)        # B×3×H×W

    # Identity-init kernel: delta function
    kernel = torch.zeros(3, 1, k, k, device=device)   # depthwise
    kernel[:, 0, k // 2, k // 2] = 1.0
    kernel.requires_grad_(True)
    opt = torch.optim.Adam([kernel], lr=5e-2)

    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=device).view(1, 1, 3, 3) / 8.0
    sy = sx.transpose(-1, -2)

    pad_k = k // 2
    pad_w = window // 2

    # Pre-compute input local extrema (constant)
    x_mx = F.max_pool2d(F.pad(X, [pad_w] * 4, mode="reflect"), window, stride=1)
    x_mn = -F.max_pool2d(F.pad(-X, [pad_w] * 4, mode="reflect"), window, stride=1)

    for it in range(steps):
        opt.zero_grad()
        Y = F.conv2d(F.pad(X, [pad_k] * 4, mode="reflect"), kernel, groups=3)

        lum = (0.2126 * Y[:, 0:1] + 0.7152 * Y[:, 1:2] + 0.0722 * Y[:, 2:3])
        gx = F.conv2d(F.pad(lum, [1] * 4, mode="reflect"), sx)
        gy = F.conv2d(F.pad(lum, [1] * 4, mode="reflect"), sy)
        sharpness = (gx ** 2 + gy ** 2).mean()

        over = F.relu(Y - x_mx - tol) ** 2 + F.relu(x_mn - tol - Y) ** 2
        overshoot = over.mean()

        content = ((Y - X) ** 2).mean()

        loss = -lam_sharp * sharpness + alpha_overshoot * overshoot + beta_content * content
        loss.backward()
        opt.step()
        if verbose and (it % 250 == 0 or it == steps - 1):
            print(f"    it={it:4d}  sharp={sharpness.item():.4f}  "
                  f"over={overshoot.item():.5f}  content={content.item():.5f}  "
                  f"loss={loss.item():.4f}")

    info = {"sharp": sharpness.item(), "overshoot": overshoot.item(),
            "content": content.item()}
    return kernel.detach().cpu().numpy(), info


def apply_learnable_kernel(rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply per-channel k×k kernel (3, 1, k, k) to uint8 H×W×3 RGB."""
    k = kernel.shape[-1]
    pad = k // 2
    X = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None] / 255.0
    K = torch.from_numpy(kernel)
    Y = F.conv2d(F.pad(X, [pad] * 4, mode="reflect"), K, groups=3)
    Y = Y.squeeze(0).permute(1, 2, 0).numpy() * 255.0
    return np.clip(Y, 0, 255).astype(np.uint8)


# ---------- pipeline ----------

def run_upscayl(in_png: Path, out_png: Path) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(UPSCAYL_BIN),
         "-i", str(in_png), "-o", str(out_png),
         "-m", str(UPSCAYL_MODELS), "-n", MODEL,
         "-s", "4", "-f", "png"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"upscayl-bin failed: "
                           f"{(proc.stderr or proc.stdout).strip()[-300:]}")
    return time.perf_counter() - t0


def save_and_upscale(rgb: np.ndarray, work_dir: Path, tag: str) -> tuple[Path, Path]:
    in_png = work_dir / f"_pre_{tag}.png"
    out_png = work_dir / f"_pre_{tag}_upscayl.png"
    Image.fromarray(rgb).save(in_png, format="PNG")
    if not out_png.exists():
        run_upscayl(in_png, out_png)
    return in_png, out_png


# ---------- chart ----------

def make_label(text: str, w: int, h: int = 60, fontsize: int = 14) -> Image.Image:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", fontsize)
    except OSError:
        font = ImageFont.load_default()
    bbox = d.multiline_textbbox((0, 0), text, font=font, spacing=2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.multiline_text(((w - tw) / 2, (h - th) / 2), text,
                     fill="black", font=font, align="center", spacing=2)
    return img


def make_chart(photo_id: str, panels: list[tuple[str, Image.Image]]) -> None:
    if not panels:
        return
    pw, ph = panels[0][1].size
    n = len(panels)
    cols = 4
    rows = (n + cols - 1) // cols
    gap = 10
    label_h = 60
    cell_w = pw + gap
    cell_h = ph + label_h + gap
    canvas_w = cols * cell_w + gap
    canvas_h = 36 + rows * cell_h + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(make_label(f"{photo_id}.jpg — sharpener preprocessing → upscayl-standard "
                            f"(crop {pw}×{ph} @ 1:1)", canvas_w, h=32, fontsize=18),
                 (0, 2))
    for i, (title, im) in enumerate(panels):
        r, c = divmod(i, cols)
        x = gap + c * cell_w
        y = 36 + r * cell_h
        canvas.paste(make_label(title, pw, h=label_h), (x, y))
        canvas.paste(im, (x, y + label_h))
    out = OUT / f"sharpen_{photo_id}.png"
    canvas.save(out, format="PNG", optimize=True)
    print(f"wrote {out}  ({canvas_w}×{canvas_h})")


# ---------- main ----------

def main() -> None:
    if not UPSCAYL_BIN.exists():
        raise SystemExit(f"upscayl-bin not found at {UPSCAYL_BIN}")

    print("loading raw renders...")
    raws = {p: render_raw_inferno(INPUT / f"{p}.jpg") for p in PHOTOS}

    # Learn best Wiener σ jointly across photos (1 param, PyTorch)
    print("\nlearning Wiener σ jointly across photos (target overshoot 20%)...")
    learned_sigma, w_info = train_wiener_sigma(list(raws.values()),
                                               target_overshoot_frac=0.20,
                                               verbose=True)
    print(f"  learned: {w_info}")

    rows = [f"{'photo':14s} {'pipeline':45s} "
            f"{'∇mean':>8s} {'∇p95':>8s} {'lapvar':>9s} {'overshoot%':>11s}",
            "-" * 100]

    for photo_id in PHOTOS:
        photo_work = WORK / photo_id
        photo_work.mkdir(parents=True, exist_ok=True)
        raw = raws[photo_id]

        print(f"\n=== {photo_id}.jpg ===")

        # Variants
        sharpened = {
            "raw (no preproc)": raw,
            "Wiener σ=0.6":   wiener_deconvolve(raw, sigma=0.6, K=1e-3),
            "Wiener σ=0.8":   wiener_deconvolve(raw, sigma=0.8, K=1e-3),
            "Wiener σ=1.0":   wiener_deconvolve(raw, sigma=1.0, K=1e-3),
            f"Wiener learned σ={learned_sigma:.2f}":
                                  wiener_deconvolve(raw, sigma=learned_sigma, K=1e-3),
            "unsharp σ=1 amt=0.5":  unsharp_mask(raw, sigma=1.0, amount=0.5),
            "unsharp σ=1 amt=1.5":  unsharp_mask(raw, sigma=1.0, amount=1.5),
        }

        # SR each variant; collect both pre and post images for the chart
        crop_panels: list[tuple[str, Image.Image]] = []
        # Add the JPEG-path baseline too for reference
        baseline_path = photo_work / "_jpgdown_upscayl-standard-4x.png"
        if baseline_path.exists():
            sr_rgb = np.array(Image.open(baseline_path).convert("RGB"))
            m = metrics_of(sr_rgb)
            rows.append(f"{photo_id:14s} {'JPEG↓ → upscayl-std (baseline)':45s} "
                        f"{m['gmean']:8.2f} {m['gp95']:8.2f} {m['lapvar']:9.1f} "
                        f"{'-':>11s}")
            (cx_f, cy_f), half = REGIONS[photo_id]
            cx, cy = int(sr_rgb.shape[1] * cx_f), int(sr_rgb.shape[0] * cy_f)
            box = (cx - half, cy - half, cx + half, cy + half)
            crop_panels.append(("JPEG↓ → upscayl-std\n(baseline)",
                               Image.fromarray(sr_rgb).crop(box)))

        for tag, pre_rgb in sharpened.items():
            slug = tag.replace(" ", "_").replace("σ", "s").replace("=", "")\
                     .replace(",", "").replace("×", "x").replace(".", "p")
            _, sr_path = save_and_upscale(pre_rgb, photo_work, slug)
            sr_rgb = np.array(Image.open(sr_path).convert("RGB"))
            m = metrics_of(sr_rgb, ref=raw)
            print(f"  {tag:30s}  ∇mean={m['gmean']:.1f}  "
                  f"lapvar={m['lapvar']:.0f}  overshoot={100*m.get('overshoot',0):.2f}%")
            rows.append(f"{photo_id:14s} "
                        f"{('raw → '+tag+' → upscayl-std'):45s} "
                        f"{m['gmean']:8.2f} {m['gp95']:8.2f} {m['lapvar']:9.1f} "
                        f"{100*m.get('overshoot',0):11.2f}")

            (cx_f, cy_f), half = REGIONS[photo_id]
            cx, cy = int(sr_rgb.shape[1] * cx_f), int(sr_rgb.shape[0] * cy_f)
            box = (cx - half, cy - half, cx + half, cy + half)
            crop_panels.append((f"raw → {tag}\n→ upscayl-std",
                               Image.fromarray(sr_rgb).crop(box)))

        rows.append("")
        make_chart(photo_id, crop_panels)

    print("\n" + "\n".join(rows))
    (OUT / "sharpen_pre.txt").write_text("\n".join(rows))
    print(f"\nwrote {OUT/'sharpen_pre.txt'}    learned σ={learned_sigma:.3f}")


if __name__ == "__main__":
    main()
