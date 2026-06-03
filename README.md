# WanVideoGeneration

Local **Wan 2.1** text-to-video and image-to-video generation on a **6 GB consumer GPU.** A single-file PyQt6 desktop app with model selection, CivitAI checkpoint + LoRA support, a Gallery with full metadata, and Copy-to-Generation reproducibility. No API costs, no per-request limits.

The 6 GB VRAM ceiling drove every design choice — bf16 throughout, sequential CPU offload, VAE tiling/slicing, and a one-time fp32→bf16 pre-conversion of the 14B model that halves its on-disk size with zero quality loss. The full engineering log is in [CHANGELOG.md](CHANGELOG.md).

## Features

- **Three model options in one UI:** Wan 2.1 T2V-1.3B (fast testing), T2V-14B (quality), I2V-14B (image-to-video). Missing models are tagged `[MISSING]` in the dropdown and disable Generate until present.
- **CivitAI checkpoints** — drop a `.safetensors` into `C:\AI_Models\Wan_Checkpoints\T2V-1.3B\` or `T2V-14B\` and it appears as `[CKP-1.3B]` / `[CKP-14B]` in the dropdown. Loaded via `WanTransformer3DModel.from_single_file()` + the base model's text-encoder / VAE / scheduler.
- **Multi-LoRA stack** with reorder, per-LoRA weight, and preview thumbnails. Uses the diffusers-native `load_lora_weights` → `set_adapters` → `fuse_lora` → `unload_lora_weights` pipeline so the fused weights work with sequential CPU offload. Filtered to the selected model's base arch (1.3B vs 14B).
- **I2V start frame** via file picker, drag-drop, Ctrl+V paste, or a cross-workspace "From Flux Gallery" button that reads from a sibling [FluxImageGeneration](https://github.com/CanGitArchive/FluxImageGeneration) workspace if one exists (override path with the `FLUX_WORKSPACE_DIR` env var; falls back to disabled if not present).
- **Gallery tab** (Outputs / Reference Frames / Best Of) with first-frame thumbnails (via ffmpeg), `QMediaPlayer` preview with seek slider, full Copy-to-Generation that restores model + checkpoint + LoRA stack + seed + prompts + start frame.
- **Time estimator** — per-arch cold-start baseline refined by per-model history (last 50 runs, normalized by `width × height × num_frames`). Live countdown.
- **Partial-download aware** — `model_exists()` checks both `model_index.json` and the absence of `.cache/huggingface/download/*.incomplete` files, so a half-pulled model correctly tags `[MISSING]`.
- **Per-video JSON sidecar** with full reproduction metadata (model, base_model, checkpoint, prompt, negative, width, height, num_frames, fps, guidance, seed, start_frame, lora_stack, elapsed, timestamp). Designed for a future "connector" tool that chains image → video → audio.

## Tech stack

PyQt6 · diffusers 0.38 · transformers · accelerate · peft · safetensors · torch 2.6.0+cu124 · imageio-ffmpeg · Python 3.12.

## Hardware

Built and tuned for an **NVIDIA RTX 4050 Laptop (6 GB VRAM), 64 GB RAM**, Windows 11. Sequential CPU offload + VAE tiling/slicing are mandatory; block-swap-to-RAM is what makes the 14B variant viable at all on consumer hardware.

Measured on this hardware:

| Model | Resolution | Time / 50-step 81-frame clip | Notes |
|---|---|---|---|
| T2V-1.3B | 832×480 | ~41 min | 2.41 GB peak VRAM |
| T2V-14B  | 832×480 | ~3.6 h  | ~44 GB RAM resident (bf16) |
| T2V-14B  | 1280×720 | **not viable** | Activations exceed 64 GB RAM → pagefile thrashes → GPU stalls. ~8 days/clip projected. 480p is the realistic 14B quality tier on this box. |

The 14B model ships fp32 on HuggingFace (transformer 54 GB + text encoder 22 GB). On 64 GB RAM the fp32→bf16 cast at load time overflows the Windows commit limit and segfaults. The fix — pre-convert to bf16 on disk via a shard-by-shard streaming cast — is documented in the CHANGELOG.

## Install

```powershell
git clone https://github.com/CanGitArchive/WanVideoGeneration.git
cd WanVideoGeneration

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The `requirements.txt` uses PyTorch's CUDA index for the `torch==2.6.0+cu124` wheel; CPU-only or other CUDA versions will need to swap the `--extra-index-url` line.

### Get the base models

Download any of these from [Wan-AI on HuggingFace](https://huggingface.co/Wan-AI) into the layout below. The app's dropdown auto-detects what's present.

```
C:\AI_Models\Wan\Wan2.1-T2V-1.3B-Diffusers\        ← base T2V 1.3B (~3 GB bf16)
C:\AI_Models\Wan\Wan2.1-T2V-14B-Diffusers\         ← base T2V 14B (recommend bf16-converting; see CHANGELOG)
C:\AI_Models\Wan\Wan2.1-I2V-14B-720P-Diffusers\    ← base I2V (optional)
C:\AI_Models\Wan_Checkpoints\T2V-1.3B\             ← drop CivitAI .safetensors here
C:\AI_Models\Wan_Checkpoints\T2V-14B\
C:\AI_Models\Wan_LoRAs\1.3B\                       ← Wan LoRAs by base arch
C:\AI_Models\Wan_LoRAs\14B\
```

The drop-in folders auto-create on first launch. Optional `.png` / `.jpg` / `.webp` preview next to a `.safetensors` becomes its dropdown icon.

### Run

```powershell
python flux_video_generator.py
```

Generated videos land in `DATA/Outputs/` as `wan_{t2v|i2v}_{seed}_{timestamp}.mp4` plus a `.thumb.jpg` first-frame thumbnail and a `.json` metadata sidecar.

## License

MIT — see [LICENSE](LICENSE).
