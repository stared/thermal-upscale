# Lab notebook

Working log for the thermal upscaling experiments. Short, explicit, dated.

## 2026-05-05 — first 14-model sweep

**Setup**
- Camera: Thermal Master P3 (192×256 sensor, ~5.83× firmware-upscaled to 1120×1494 JPEG).
- Pipeline per photo: JPEG → Lanczos → 192×256 → upscayl-bin -s 4 → 768×1024 PNG.
- Backend: `upscayl-bin` (Mach-O universal) from `/Applications/Upscayl.app/Contents/Resources/`.
- Custom NCNN models cached in `~/.cache/thermal-upscale/models/` (`.param` + `.bin` pairs from `github.com/upscayl/custom-models`).
- Test images: `1777804010136` (cups/shisha), `1777733165452` (campfire), `1771110170401` (person/dance).
- Outputs: `examples/comparison_{full,crop}_<id>.png` and `examples/artifact_<id>_<region>.png`.

**Speed (consistent across all three photos)**
- ~0.4s: `upscayl-lite-4x`, `RealESRGAN_General_x4_v3`, `_WDN_`, `digital-art-4x`
- ~1.7–1.9s: everything else, with `upscayl-standard-4x` reproducibly slowest.

**Verdict (after user review of crops + artifact panels)**

| model | verdict | notes |
|---|---|---|
| `upscayl-standard-4x` | **TOP** | reference. faithful, no halo amplification, slowest. |
| `4xNomos8kSC` | **TOP** | clean & sharp without micro-contrast push. only real alternative to Standard. |
| `RealESRGAN_General_x4_v3` | **TOP** | soft but faithful; preserves structure, doesn't hallucinate. fast. |
| `upscayl-lite-4x` | KEEP (worse than std) | softer; OK fallback. |
| `high-fidelity-4x` | KEEP (worse than std) | ≈standard, slightly softer. |
| `RealESRGAN_General_WDN_x4_v3` | KEEP (worse than non-WDN) | denoised → over-soft. |
| `4xHFA2k` | KEEP (different) | painterly, smooths embers but doesn't fabricate. |
| `digital-art-4x` | KEEP (different aesthetics) | smoother gradients, distinct look. |
| `ultramix-balanced-4x` | **DROP** | redundant with std/hi-fi. |
| `remacri-4x` | **DROP** | aggressive halo amplification (esp. `straw-edge`). |
| `ultrasharp-4x` | **DROP** | hallucinates: invents stripe patterns on cup ridges; warm streaks in `skirt-chain` not in source. |
| `4x_NMKD-Siax_200k` | **DROP** | introduces texture/grain not in source. |
| `4x_NMKD-Superscale-SP_178000_G` | **DROP** | same family as Siax. |
| `4xLSDIRplusC` | **DROP** | looks close to Nomos but introduces faint micro-texture. user-flagged as "BAD, not same league as Nomos". |

**Most discriminating regions** (in order)
1. **`belt-ornament` (person image) — canonical hardest test.** Hexagonal spokes ~1px in 192×256; reveals both hallucination (sharp models invent extra spokes) and over-smoothing (smoothing flattens the pattern).
2. `straw-edge` (cups) — exposes halo amplification + fabricated bowl-ridge stripes.
3. `skirt-chain` (person) — exposes hallucinated warm streaks (ultrasharp confirmed).
4. `ember-texture` (campfire) — over-smoothing test; HFA2k flattens, RealESRGAN softens.

**Divergence between my initial read and user verdict**
- I dropped `RealESRGAN_General_x4_v3` and `4xHFA2k` for being "soft / over-smoothing"; user kept them.
- I kept `4x_NMKD-Siax`, `4x_NMKD-Superscale`, `4xLSDIRplusC` for being "clean & sharp"; user dropped them.
- **Calibration:** user values *cleanness and faithfulness over preserved high-frequency texture*. Slight softness is acceptable; any model that introduces micro-contrast / texture not in source is rejected, even if subtle. Aesthetic variety (e.g. `digital-art`) is welcome as a separate category, not a replacement for Standard.

**Key finding — JPEG halo problem**
- Camera firmware bakes a bright edge halo around hot/cold contours into the JPEG (most visible on `upper-cup-rim`).
- Surviving the downscale, this halo is *preserved* by faithful models, *amplified* by aggressive ones (ultrasharp, remacri), *softened* by smooth ones (HFA2k, digital-art).
- **No upscaler can remove it** — only attenuate. Real fix has to come from upstream: extract the raw 192×256 uint16 from APP3 segments and apply our own colormap. That bypasses the JPEG halo entirely.

## Models pool (going forward)

Active set for any future comparison (8 models, encoded in both scripts):
- `upscayl-standard-4x` — default
- `4xNomos8kSC` — primary alternative
- `RealESRGAN_General_x4_v3` — fast option, soft-but-faithful
- `4xHFA2k`, `digital-art-4x` — aesthetic variants
- `upscayl-lite-4x`, `high-fidelity-4x`, `RealESRGAN_General_WDN_x4_v3` — kept as worse-than-best fallbacks; will likely drop after one more sweep.

Dropped, do not re-run:
- `ultrasharp-4x`, `remacri-4x`, `ultramix-balanced-4x`, `4x_NMKD-Siax_200k`, `4x_NMKD-Superscale-SP_178000_G`, `4xLSDIRplusC`

## Open experiments / next

**Constraint:** the upscaling itself is **one** neural network. Not a cascade, not a two-stage refinement, not two models stitched together. No classical pre/post filters either (Gaussian, unsharp, CAS, denoising). Fine-tuned weights are fair game. Lanczos as the *downscale to native sensor size* stays — that's input prep, not part of the SR step.

Given that, the only remaining levers are: **(a) which single network** and **(b) what input that network sees**.

- [ ] **Raw-mode pipeline.** Extract APP3 uint16 → normalize → apply colormap → 192×256 PNG → upscayl-bin Standard. Pure input-side change; same single network. Should remove the camera-firmware JPEG halo entirely. Compare colormap variants (ironbow / inferno / turbo) since the colormap *is* what the SR network sees.
- [ ] **A/B Standard vs Nomos8kSC** on ~10 more scenes (3-image sample isn't decisive). Diverse subjects: people, indoor objects, outdoor warm/cool, low-thermal-contrast frames.
- [ ] **Thermal-specific fine-tune.** Out of scope unless we collect a P3 dataset and have GPU time. Park.
- [ ] **CLI** (only after the above settle): `thermal-upscale INPUT [-o OUT] [--from-raw] [--colormap NAME] [--model NAME]`. Default `upscayl-standard-4x`. Single-file or directory input.

## Tooling notes
- `upscayl-bin -i in -o out -m <models_dir> -n <model> -s 4 -f png` — verified on all 14 models. Works from PNG input; PNG round-trip avoids JPEG re-compression on the 192×256 intermediate.
- The `upscayl/custom-models` repo URL pattern is `https://github.com/upscayl/custom-models/raw/main/models/<name>.{param,bin}`. Stable; downloaded 7 models without rate-limiting.
