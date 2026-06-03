r"""
Flux Video Generator — sibling PyQt6 app to flux_image_generator.py.

Local Wan 2.1 text-to-video and image-to-video on a 6 GB GPU using the proven
diffusers stack: WanPipeline / WanImageToVideoPipeline + sequential CPU offload
+ VAE tiling/slicing. Same dark purple/blue theme, worker pattern, and Gallery
pattern as the image gen.

Three models exposed as one UI:
  - Wan 2.1 T2V-1.3B   (fast testing, 480p)
  - Wan 2.1 T2V-14B    (quality T2V, 480p / 720p)
  - Wan 2.1 I2V-14B    (image-to-video, takes a start frame)

Missing models show [MISSING] in the dropdown and disable Generate until the
local dir exists. Settings persist to DATA/settings.json, history to
DATA/gen_history.json, outputs to DATA/Outputs/ with full .json metadata
sidecars (connector-ready). The I2V start-frame picker reads from the sibling
FluxImageGeneration workspace's DATA/Outputs and DATA/BestOf for cross-app
integration.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reduce 6 GB fragmentation; harmless on platforms that don't support it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PyQt6.QtCore import (
    Qt, QThread, QTimer, QElapsedTimer, QSize, QUrl, pyqtSignal, QPoint,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QIcon, QKeySequence, QShortcut, QFont, QDragEnterEvent,
    QDropEvent,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QPlainTextEdit, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFrame, QTabWidget, QFileDialog, QMessageBox,
    QProgressBar, QSlider, QSplitter, QListWidget, QListWidgetItem, QSizePolicy,
    QStatusBar, QStackedWidget, QStyle,
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget


# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------

APP_VERSION = "0.1"

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent

# Per-workspace DATA root (renamed from the old combined "DATA/FluxImageGenerator/"
# layout when this workspace split off from FluxImageGenerator on 2026-05-21).
DATA_DIR = APP_DIR / "DATA"
VIDEOS_DIR = DATA_DIR / "Outputs"
VIDEO_REF_FRAMES_DIR = DATA_DIR / "ReferenceImages"
VIDEO_BESTOF_DIR = DATA_DIR / "BestOf"
SETTINGS_FILE = DATA_DIR / "settings.json"
HISTORY_FILE = DATA_DIR / "gen_history.json"

# Cross-workspace integration: the I2V "From Flux Gallery" picker reads from a
# sibling FluxImageGeneration workspace if one exists. By default we look for
# it at `../FluxImageGeneration` (i.e. a folder next to this one). Override
# with the FLUX_WORKSPACE_DIR env var. The picker disables gracefully if the
# path doesn't exist.
FLUX_WORKSPACE_DIR = Path(
    os.environ.get("FLUX_WORKSPACE_DIR", APP_DIR.parent / "FluxImageGeneration")
)
FLUX_OUTPUTS_DIR = FLUX_WORKSPACE_DIR / "DATA" / "Outputs"
FLUX_BESTOF_DIR = FLUX_WORKSPACE_DIR / "DATA" / "BestOf"

WAN_BASE = Path("C:/AI_Models/Wan")
WAN_CHECKPOINTS_DIR = Path("C:/AI_Models/Wan_Checkpoints")  # T2V-1.3B/, T2V-14B/
WAN_LORAS_DIR = Path("C:/AI_Models/Wan_LoRAs")              # 1.3B/, 14B/

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv"}

# Wan default constants — proven recipe from tools/_wan_spike_t2v_1_3b.py.
DEFAULT_NEGATIVE = (
    "low quality, blurry, distorted, watermark, text, deformed face, ugly, "
    "oversaturated, jpeg artifacts, static, frozen"
)

# Cold-start fallback rate (s/step) — by base arch, overridden by real history.
# Both measured at 832×480×81f on the RTX 4050 6 GB / 64 GB RAM box:
#   1.3B: 2425 s / 50 steps = 48.5 s/step
#   14B:  ~244 s/step (2026-05-22, bf16 model, sequential CPU offload)
# NOTE: this rate is the *compute* rate at 480p. The 14B model thrashes RAM at
# 720p (44 GB model + 720p activations > 64 GB) — there the real rate is ~100x
# worse and the estimate below is meaningless. See CLAUDE.md "Roadmap" #2.
COLD_START_SPS_BY_ARCH = {
    "1.3B": 48.5,
    "14B":  244.0,
}


# ---------------------------------------------------------------------------
# Wan model registry — base models + discovered checkpoints (runtime)
# ---------------------------------------------------------------------------

# Base model config. Each entry is keyed by display name. `arch` is "1.3B" or
# "14B" (drives both LoRA filtering and the cold-start s/step rate); `kind`
# is "t2v" or "i2v" (drives pipeline class + start-frame UI).
WAN_BASE_MODELS: Dict[str, Dict[str, Any]] = {
    "Wan 2.1 T2V-1.3B": {
        "path": WAN_BASE / "Wan2.1-T2V-1.3B-Diffusers",
        "arch": "1.3B",
        "kind": "t2v",
        "resolutions": [(832, 480, "832×480 (480p)")],
        "default_fps": 15,
        "default_steps": 50,
        "default_guidance": 5.0,
        "default_frames": 81,
    },
    "Wan 2.1 T2V-14B": {
        "path": WAN_BASE / "Wan2.1-T2V-14B-Diffusers",
        "arch": "14B",
        "kind": "t2v",
        "resolutions": [
            (832, 480, "832×480 (480p)"),
            (1280, 720, "1280×720 (720p)"),
        ],
        "default_fps": 16,
        "default_steps": 50,
        "default_guidance": 5.0,
        "default_frames": 81,
    },
    "Wan 2.1 I2V-14B (720p)": {
        "path": WAN_BASE / "Wan2.1-I2V-14B-720P-Diffusers",
        "arch": "14B",
        "kind": "i2v",
        "resolutions": [(1280, 720, "1280×720 (720p)")],
        "default_fps": 16,
        "default_steps": 50,
        "default_guidance": 5.0,
        "default_frames": 81,
    },
    "Wan 2.1 I2V-14B (480p)": {
        "path": WAN_BASE / "Wan2.1-I2V-14B-480P-Diffusers",
        "arch": "14B",
        "kind": "i2v",
        "resolutions": [(832, 480, "832×480 (480p)")],
        "default_fps": 16,
        "default_steps": 50,
        "default_guidance": 5.0,
        "default_frames": 81,
    },
}

# Maps checkpoint subfolder name -> the base model whose architecture it sits on.
# T2V checkpoints only for now (I2V community fine-tunes are rare).
CHECKPOINT_SUBFOLDERS: Dict[str, str] = {
    "T2V-1.3B": "Wan 2.1 T2V-1.3B",
    "T2V-14B":  "Wan 2.1 T2V-14B",
}

# Runtime model registry — base + discovered checkpoints, rebuilt by
# rebuild_model_registry(). Each entry:
#   "kind":           "base" | "checkpoint"
#   "pipeline_kind":  "t2v" | "i2v"  (from base)
#   "arch":           "1.3B" | "14B" (from base)
#   "base_name":      key into WAN_BASE_MODELS to source pipeline + defaults
#   "checkpoint_path": Path | None
#   "resolutions":    forwarded from base
#   "default_*":      forwarded from base
_MODEL_ENTRIES: Dict[str, Dict[str, Any]] = {}


def _make_base_entry(name: str) -> Dict[str, Any]:
    info = WAN_BASE_MODELS[name]
    return {
        "kind": "base",
        "pipeline_kind": info["kind"],
        "arch": info["arch"],
        "base_name": name,
        "base_path": info["path"],
        "checkpoint_path": None,
        "resolutions": info["resolutions"],
        "default_fps": info["default_fps"],
        "default_steps": info["default_steps"],
        "default_guidance": info["default_guidance"],
        "default_frames": info["default_frames"],
    }


def _make_checkpoint_entry(file: Path, base_name: str) -> Dict[str, Any]:
    info = WAN_BASE_MODELS[base_name]
    return {
        "kind": "checkpoint",
        "pipeline_kind": info["kind"],  # T2V for now
        "arch": info["arch"],
        "base_name": base_name,
        "base_path": info["path"],
        "checkpoint_path": file,
        "resolutions": info["resolutions"],
        "default_fps": info["default_fps"],
        "default_steps": info["default_steps"],
        "default_guidance": info["default_guidance"],
        "default_frames": info["default_frames"],
    }


def scan_checkpoints() -> List[Tuple[str, Path, str]]:
    """Return list of (display_name, file_path, base_name) for every checkpoint
    found under WAN_CHECKPOINTS_DIR/<subfolder>/*.safetensors."""
    found: List[Tuple[str, Path, str]] = []
    for sub, base_name in CHECKPOINT_SUBFOLDERS.items():
        sub_dir = WAN_CHECKPOINTS_DIR / sub
        if not sub_dir.exists():
            continue
        for f in sorted(sub_dir.glob("*.safetensors")):
            arch = WAN_BASE_MODELS[base_name]["arch"]
            display = f"[CKP-{arch}] {f.stem}"
            found.append((display, f, base_name))
    return found


def scan_loras_for_arch(arch: str) -> List[Path]:
    """Return list of .safetensors files under WAN_LORAS_DIR/<arch>/."""
    sub = WAN_LORAS_DIR / arch
    if not sub.exists():
        return []
    return sorted(sub.glob("*.safetensors"))


def rebuild_model_registry() -> None:
    """Repopulate _MODEL_ENTRIES from base models + scanned checkpoints."""
    _MODEL_ENTRIES.clear()
    for name in WAN_BASE_MODELS:
        _MODEL_ENTRIES[name] = _make_base_entry(name)
    for display, file, base_name in scan_checkpoints():
        _MODEL_ENTRIES[display] = _make_checkpoint_entry(file, base_name)


def get_entry(name: str) -> Optional[Dict[str, Any]]:
    return _MODEL_ENTRIES.get(name)


def _has_incomplete_downloads(model_dir: Path) -> bool:
    dl_cache = model_dir / ".cache" / "huggingface" / "download"
    if not dl_cache.exists():
        return False
    try:
        for _ in dl_cache.rglob("*.incomplete"):
            return True
    except Exception:
        pass
    return False


def model_exists(name: str) -> bool:
    """True if everything needed to generate with this entry is on disk."""
    entry = _MODEL_ENTRIES.get(name)
    if not entry:
        return False
    base_path: Path = entry["base_path"]
    # Base model dir must be present and not mid-download.
    if not base_path.exists() or not (base_path / "model_index.json").exists():
        return False
    if _has_incomplete_downloads(base_path):
        return False
    # For checkpoints, the .safetensors file itself must exist.
    if entry["kind"] == "checkpoint":
        ckpt = entry["checkpoint_path"]
        if not ckpt or not ckpt.exists():
            return False
    return True


def model_display_label(name: str) -> str:
    return name if model_exists(name) else f"[MISSING] {name}"


def preview_for_file(file: Path) -> Optional[Path]:
    """Return matching .png/.jpg/.jpeg/.webp preview for `file` if present."""
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = file.with_suffix(ext)
        if p.exists():
            return p
    return None


# Populate the registry at import time so the module is self-sufficient — a
# headless consumer (e.g. the future connector app) can construct and run a
# GenerateWorker without first building a MainWindow. MainWindow.__init__ still
# calls rebuild_model_registry() again, which just re-scans for newly added
# checkpoints; the Refresh Models button does the same on demand.
rebuild_model_registry()


# ---------------------------------------------------------------------------
# Settings & history persistence
# ---------------------------------------------------------------------------

def ensure_data_dirs() -> None:
    for p in (DATA_DIR, VIDEOS_DIR, VIDEO_REF_FRAMES_DIR, VIDEO_BESTOF_DIR):
        p.mkdir(parents=True, exist_ok=True)
    # Pre-create the model drop-in folders so the user has obvious targets.
    for sub in CHECKPOINT_SUBFOLDERS:
        (WAN_CHECKPOINTS_DIR / sub).mkdir(parents=True, exist_ok=True)
    for arch in ("1.3B", "14B"):
        (WAN_LORAS_DIR / arch).mkdir(parents=True, exist_ok=True)


def load_settings() -> Dict[str, Any]:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_settings(settings: Dict[str, Any]) -> None:
    try:
        SETTINGS_FILE.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def _load_history() -> List[Dict[str, Any]]:
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def log_generation_time(
    model: str, width: int, height: int, num_frames: int, steps: int,
    elapsed: float, has_start_frame: bool,
) -> None:
    history = _load_history()
    history.append({
        "model": model,
        "width": int(width),
        "height": int(height),
        "frames": int(num_frames),
        "steps": int(steps),
        "elapsed": round(float(elapsed), 1),
        "has_start_frame": bool(has_start_frame),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    history = history[-50:]
    try:
        HISTORY_FILE.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def _estimate_seconds_per_step(model: str, width: int, height: int,
                                num_frames: int) -> Optional[float]:
    """Average s/step across the most recent matching runs, normalized by pixels.
    Returns None if no history exists; caller falls back to COLD_START_SPS."""
    history = _load_history()
    target_px = max(1, width * height * num_frames)
    rates: List[float] = []
    for h in reversed(history):
        if h.get("model") != model:
            continue
        steps = h.get("steps") or 0
        elapsed = h.get("elapsed") or 0.0
        h_px = (h.get("width") or 0) * (h.get("height") or 0) * (h.get("frames") or 0)
        if steps <= 0 or elapsed <= 0 or h_px <= 0:
            continue
        # Normalize the historic per-step rate to the requested pixel volume.
        normalized_sps = (elapsed / steps) * (target_px / h_px)
        rates.append(normalized_sps)
        if len(rates) >= 5:
            break
    if not rates:
        return None
    return sum(rates) / len(rates)


def estimate_total_seconds(model: str, width: int, height: int,
                            num_frames: int, steps: int) -> float:
    sps = _estimate_seconds_per_step(model, width, height, num_frames)
    if sps is None:
        # Cold start: scale the spike's 480p baseline by pixel volume.
        entry = _MODEL_ENTRIES.get(model)
        arch = entry["arch"] if entry else "1.3B"
        base_sps = COLD_START_SPS_BY_ARCH.get(arch, 60.0)
        baseline_px = 832 * 480 * 81
        target_px = max(1, width * height * num_frames)
        sps = base_sps * (target_px / baseline_px)
    return sps * steps


def format_eta(total_seconds: float) -> str:
    if total_seconds < 60:
        return f"~{int(total_seconds)}s"
    if total_seconds < 3600:
        m = int(total_seconds // 60)
        s = int(total_seconds % 60)
        return f"~{m}m {s}s"
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    return f"~{h}h {m}m"


# ---------------------------------------------------------------------------
# Video output: save metadata, extract first-frame thumbnail
# ---------------------------------------------------------------------------

def write_sidecar(video_path: Path, meta: Dict[str, Any]) -> None:
    side = video_path.with_suffix(".json")
    try:
        side.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    except Exception:
        pass


def read_sidecar(video_path: Path) -> Optional[Dict[str, Any]]:
    side = video_path.with_suffix(".json")
    if not side.exists():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def _ffmpeg_binary() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def extract_thumbnail(video_path: Path, thumb_path: Path) -> bool:
    """Pull the first frame of `video_path` as JPEG via ffmpeg. Best-effort."""
    if thumb_path.exists():
        return True
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg:
        return False
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video_path), "-frames:v", "1",
             "-q:v", "3", str(thumb_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return thumb_path.exists()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# GenerateWorker — runs the Wan pipeline in a background thread
# ---------------------------------------------------------------------------

class GenerateWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)            # (step, total_steps)
    video_done = pyqtSignal(str, int, dict)    # (path, seed, meta)
    completed = pyqtSignal(float)              # elapsed seconds
    failed = pyqtSignal(str)

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(tb)
            self.failed.emit(str(e))

    def _run(self) -> None:
        cfg = self.config
        model_name: str = cfg["model"]
        entry = get_entry(model_name)
        if not entry:
            raise RuntimeError(f"Unknown model: {model_name}")
        pipeline_kind = entry["pipeline_kind"]
        base_path: Path = entry["base_path"]
        checkpoint_path: Optional[Path] = entry["checkpoint_path"]

        if not model_exists(model_name):
            raise RuntimeError(
                f"Model missing or incomplete on disk:\n  base: {base_path}"
                + (f"\n  checkpoint: {checkpoint_path}" if checkpoint_path else "")
            )

        # Heavy imports happen here so the UI starts up snappy.
        import torch
        from diffusers.utils import export_to_video
        from diffusers import WanTransformer3DModel
        if pipeline_kind == "i2v":
            from diffusers import WanImageToVideoPipeline as PipelineClass
        else:
            from diffusers import WanPipeline as PipelineClass

        self.log.emit(
            f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
            f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
        )

        # ------------------------------------------------------------------
        # Load pipeline (CPU offload + VAE tiling/slicing — proven 6 GB stack)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        if entry["kind"] == "checkpoint" and checkpoint_path is not None:
            self.log.emit(
                f"loading checkpoint {checkpoint_path.name} (single_file) "
                f"with base text-encoder/VAE from {base_path} (bf16)..."
            )
            try:
                transformer = WanTransformer3DModel.from_single_file(
                    str(checkpoint_path),
                    config=str(base_path / "transformer"),
                    torch_dtype=torch.bfloat16,
                )
            except Exception as e:
                # Same diagnostic pattern as the Flux checkpoint loader.
                try:
                    from safetensors import safe_open
                    with safe_open(str(checkpoint_path), framework="pt") as st:
                        keys = list(st.keys())
                    prefixes = sorted({k.split('.')[0] for k in keys})[:20]
                    self.log.emit(
                        f"checkpoint load failed: {len(keys)} keys, "
                        f"top-level prefixes: {prefixes}"
                    )
                except Exception:
                    pass
                raise RuntimeError(
                    f"from_single_file failed for {checkpoint_path.name}: {e}"
                )
            pipe = PipelineClass.from_pretrained(
                str(base_path), transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
        else:
            self.log.emit(f"loading {model_name} from {base_path} (bf16)...")
            pipe = PipelineClass.from_pretrained(str(base_path),
                                                  torch_dtype=torch.bfloat16)
        self.log.emit(f"loaded in {time.perf_counter() - t0:.0f}s")

        # ------------------------------------------------------------------
        # Apply LoRA stack BEFORE CPU offload (so offload hooks see fused
        # weights). Pattern mirrors flux_image_generator.apply_lora_stack:
        # load_lora_weights per file → set_adapters → fuse_lora → unload.
        # ------------------------------------------------------------------
        lora_stack: List[Dict[str, Any]] = cfg.get("lora_stack") or []
        if lora_stack:
            self._apply_lora_stack(pipe, lora_stack)

        self.log.emit("attaching sequential CPU offload + VAE tiling/slicing...")
        pipe.enable_sequential_cpu_offload()
        if hasattr(pipe, "vae"):
            for fn_name in ("enable_tiling", "enable_slicing"):
                try:
                    getattr(pipe.vae, fn_name)()
                except Exception as e:
                    self.log.emit(f"  (vae.{fn_name} not available: {e})")

        if self._cancel_requested:
            self._cleanup(pipe)
            self.failed.emit("Cancelled before generation.")
            return

        # ------------------------------------------------------------------
        # Prepare generation kwargs
        # ------------------------------------------------------------------
        width = int(cfg["width"])
        height = int(cfg["height"])
        num_frames = int(cfg["num_frames"])
        fps = int(cfg["fps"])
        steps = int(cfg["num_inference_steps"])
        guidance = float(cfg["guidance_scale"])
        seed = int(cfg["seed"])
        prompt = cfg["prompt"]
        neg = cfg.get("negative_prompt", "") or ""
        start_frame_path = cfg.get("start_frame")

        gen = torch.Generator(device="cpu").manual_seed(seed)

        pipe_kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=neg,
            height=height, width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        )

        if pipeline_kind == "i2v":
            if not start_frame_path:
                self._cleanup(pipe)
                raise RuntimeError("I2V model selected but no start frame provided.")
            from PIL import Image
            img = Image.open(start_frame_path).convert("RGB")
            img = _fit_image(img, width, height)
            pipe_kwargs["image"] = img
            self.log.emit(f"start frame: {start_frame_path} -> {img.size}")

        self.log.emit(
            f"generating: {width}x{height}, {num_frames} frames @ {fps} fps "
            f"(~{num_frames / fps:.1f}s), steps={steps}, guidance={guidance}, "
            f"seed={seed}"
        )

        # Step progress + cooperative cancel via callback_on_step_end.
        def _cb(p, step_idx, timestep, callback_kwargs):
            self.progress.emit(int(step_idx) + 1, steps)
            if self._cancel_requested:
                # Raising aborts the pipeline call cleanly.
                raise _CancelException()
            return callback_kwargs

        pipe_kwargs["callback_on_step_end"] = _cb
        # diffusers expects the list of tensors the callback may want to read.
        pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        # ------------------------------------------------------------------
        # Run
        # ------------------------------------------------------------------
        t_gen = time.perf_counter()
        try:
            result = pipe(**pipe_kwargs)
        except _CancelException:
            self._cleanup(pipe)
            self.failed.emit("Cancelled.")
            return
        elapsed = time.perf_counter() - t_gen
        self.log.emit(
            f"generation complete in {elapsed:.0f}s "
            f"({elapsed / steps:.1f}s/step, {elapsed / num_frames:.1f}s/frame)"
        )

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        frames = result.frames[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        kind_tag = "i2v" if pipeline_kind == "i2v" else "t2v"
        stem = f"wan_{kind_tag}_{seed}_{ts}"
        out_mp4 = VIDEOS_DIR / f"{stem}.mp4"
        export_to_video(frames, str(out_mp4), fps=fps)
        self.log.emit(f"wrote {out_mp4} ({out_mp4.stat().st_size / 1e6:.1f} MB)")

        # First-frame thumbnail next to the mp4 (gallery uses this).
        extract_thumbnail(out_mp4, out_mp4.with_suffix(".thumb.jpg"))

        meta = {
            "model": model_name,
            "base_model": entry["base_name"],
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "prompt": prompt,
            "negative_prompt": neg,
            "width": width, "height": height,
            "num_frames": num_frames, "fps": fps,
            "duration_seconds": round(num_frames / fps, 3),
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "start_frame": str(start_frame_path) if start_frame_path else None,
            "lora_stack": [
                {"file": str(item.get("file")), "weight": float(item.get("weight", 1.0))}
                for item in (cfg.get("lora_stack") or [])
            ],
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_sidecar(out_mp4, meta)

        log_generation_time(model_name, width, height, num_frames, steps,
                            elapsed, has_start_frame=bool(start_frame_path))

        self.video_done.emit(str(out_mp4), seed, meta)
        self.completed.emit(elapsed)

        self._cleanup(pipe)

    def _apply_lora_stack(self, pipe, stack: List[Dict[str, Any]]) -> None:
        """Load each LoRA via diffusers' native loader, set adapter weights,
        fuse into the base, then unload. Matches the proven Flux pattern so
        sequential CPU offload (run later) only sees plain modules."""
        adapter_names: List[str] = []
        weights: List[float] = []
        for i, item in enumerate(stack):
            file = Path(item.get("file", ""))
            weight = float(item.get("weight", 1.0))
            if not file.exists():
                self.log.emit(f"  LoRA[{i}] MISSING -> skipping: {file}")
                continue
            adapter_name = f"lora_{i}_{file.stem[:24]}"
            self.log.emit(f"  loading LoRA[{i}] @ {weight:.2f}: {file.name}")
            try:
                pipe.load_lora_weights(
                    str(file.parent), weight_name=file.name,
                    adapter_name=adapter_name,
                )
                adapter_names.append(adapter_name)
                weights.append(weight)
            except Exception as e:
                self.log.emit(f"  LoRA[{i}] load failed: {e}")
        if not adapter_names:
            self.log.emit("  no LoRAs loaded")
            return
        try:
            pipe.set_adapters(adapter_names, adapter_weights=weights)
            pipe.fuse_lora()
            pipe.unload_lora_weights()
            self.log.emit(f"  fused {len(adapter_names)} LoRA(s) into the base")
        except Exception as e:
            self.log.emit(f"  LoRA fuse/unload failed: {e}")

    def _cleanup(self, pipe) -> None:
        try:
            del pipe
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


class _CancelException(Exception):
    """Raised inside the step callback to abort the pipeline cleanly."""


def _fit_image(img, target_w: int, target_h: int):
    """Center-crop + resize a PIL image to (target_w, target_h)."""
    from PIL import Image
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    tgt_ratio = target_w / target_h
    if src_ratio > tgt_ratio:
        # Source is wider — crop sides.
        new_w = int(src_h * tgt_ratio)
        x0 = (src_w - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, src_h))
    elif src_ratio < tgt_ratio:
        # Source is taller — crop top/bottom.
        new_h = int(src_w / tgt_ratio)
        y0 = (src_h - new_h) // 2
        img = img.crop((0, y0, src_w, y0 + new_h))
    return img.resize((target_w, target_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------

class _PreviewLabel(QLabel):
    """QLabel that re-renders its pixmap on resize to stay sharp."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pix: Optional[QPixmap] = None
        self.setObjectName("PreviewFrame")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self._double_click_cb = None

    def set_image(self, path: Optional[str]) -> None:
        if not path or not Path(path).exists():
            self._pix = None
            self.clear()
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            self._pix = None
            self.clear()
            return
        self._pix = pix
        self._rerender()

    def set_double_click_handler(self, cb) -> None:
        self._double_click_cb = cb

    def mouseDoubleClickEvent(self, ev):
        if self._double_click_cb:
            self._double_click_cb()
        super().mouseDoubleClickEvent(ev)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rerender()

    def _rerender(self) -> None:
        if not self._pix:
            return
        scaled = self._pix.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class VideoPlayerWidget(QWidget):
    """Compact video preview: QVideoWidget + play/pause + seek slider."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(480, 270)
        self.video_widget.setStyleSheet(
            "background:#0f1322;border:1px solid #3a4265;border-radius:10px;"
        )
        lay.addWidget(self.video_widget, 1)

        ctrls = QHBoxLayout()
        ctrls.setSpacing(8)

        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPlay))
        self.play_btn.setFixedWidth(46)
        self.play_btn.clicked.connect(self._toggle_play)
        ctrls.addWidget(self.play_btn)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self._on_slider_moved)
        ctrls.addWidget(self.position_slider, 1)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setMinimumWidth(90)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrls.addWidget(self.time_label)

        lay.addLayout(ctrls)

        self.audio_output = QAudioOutput()
        self.audio_output.setMuted(True)  # Wan output has no audio anyway.
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_state)

    def set_source(self, path: Optional[str], autoplay: bool = True) -> None:
        if not path:
            self.player.setSource(QUrl())
            self.position_slider.setRange(0, 0)
            self.time_label.setText("0:00 / 0:00")
            return
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        if autoplay:
            self.player.play()

    def stop(self) -> None:
        self.player.stop()

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_slider_moved(self, pos: int) -> None:
        self.player.setPosition(pos)

    def _on_position(self, pos: int) -> None:
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(pos)
        self.time_label.setText(
            f"{_fmt_ms(pos)} / {_fmt_ms(self.player.duration())}"
        )

    def _on_duration(self, dur: int) -> None:
        self.position_slider.setRange(0, dur)
        self.time_label.setText(
            f"{_fmt_ms(self.player.position())} / {_fmt_ms(dur)}"
        )

    def _on_state(self, state) -> None:
        icon = QStyle.StandardPixmap.SP_MediaPause if state == QMediaPlayer.PlaybackState.PlayingState else QStyle.StandardPixmap.SP_MediaPlay
        self.play_btn.setIcon(self.style().standardIcon(icon))


def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    m = s // 60
    s = s % 60
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"FluxVideoGenerator V{APP_VERSION}")
        self.resize(1500, 950)
        ensure_data_dirs()
        rebuild_model_registry()

        self.worker: Optional[GenerateWorker] = None
        self._start_frame_path: Optional[str] = None
        self._current_video_path: Optional[str] = None
        # Each item: {"file": str, "weight": float, "arch": "1.3B" | "14B"}.
        self._lora_stack: List[Dict[str, Any]] = []

        self._elapsed_timer = QElapsedTimer()
        self._tick_timer = QTimer()
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._update_elapsed_display)

        self.build_ui()
        self.apply_style()
        self.load_saved_settings()
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def build_ui(self) -> None:
        central = QTabWidget()
        central.setObjectName("TopTabs")
        self.setCentralWidget(central)

        # Generate tab
        gen_page = QWidget()
        gen_layout = QHBoxLayout(gen_page)
        gen_layout.setContentsMargins(10, 10, 10, 10)
        gen_layout.setSpacing(10)

        self.left_tabs = QTabWidget()
        self.left_tabs.setObjectName("LeftTabs")
        self.left_tabs.setFixedWidth(400)
        self.left_tabs.addTab(self._build_model_tab(), "Model")
        self.left_tabs.addTab(self._build_settings_tab(), "Settings")
        gen_layout.addWidget(self.left_tabs)

        gen_layout.addWidget(self._build_generate_main_area(), 1)

        central.addTab(gen_page, "Generate")

        # Gallery tab
        central.addTab(self._build_gallery_tab(), "Gallery")

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready.")

    def _build_model_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # Model dropdown + refresh row.
        model_row = QHBoxLayout()
        model_row.addWidget(_section_label("Model"), 1)
        self.refresh_models_btn = QPushButton("Refresh")
        self.refresh_models_btn.setToolTip(
            "Rescan checkpoints + LoRAs on disk without restarting."
        )
        self.refresh_models_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(self.refresh_models_btn)
        lay.addLayout(model_row)

        self.model_combo = QComboBox()
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_change)
        lay.addWidget(self.model_combo)

        self.model_status_label = QLabel("")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setStyleSheet("color:#9aa3c8;font-size:11px;")
        lay.addWidget(self.model_status_label)

        lay.addSpacing(6)
        lay.addWidget(_section_label("Resolution"))
        self.resolution_combo = QComboBox()
        lay.addWidget(self.resolution_combo)

        # LoRA stack panel.
        lay.addSpacing(6)
        self.lora_panel = QFrame()
        self.lora_panel.setObjectName("Panel")
        lp = QVBoxLayout(self.lora_panel)
        lp.setContentsMargins(8, 8, 8, 8)
        lp.setSpacing(6)
        lp.addWidget(_section_label("LoRAs (filtered by base arch)"))

        add_row = QHBoxLayout()
        self.lora_add_combo = QComboBox()
        self.lora_add_combo.setMinimumWidth(180)
        add_row.addWidget(self.lora_add_combo, 1)
        self.lora_add_btn = QPushButton("Add")
        self.lora_add_btn.clicked.connect(self._on_lora_add)
        add_row.addWidget(self.lora_add_btn)
        lp.addLayout(add_row)

        self.lora_list = QListWidget()
        self.lora_list.setMinimumHeight(120)
        self.lora_list.setMaximumHeight(220)
        lp.addWidget(self.lora_list)

        stack_btn_row = QHBoxLayout()
        self.lora_up_btn = QPushButton("▲")
        self.lora_up_btn.setFixedWidth(36)
        self.lora_up_btn.clicked.connect(lambda: self._on_lora_reorder(-1))
        stack_btn_row.addWidget(self.lora_up_btn)
        self.lora_down_btn = QPushButton("▼")
        self.lora_down_btn.setFixedWidth(36)
        self.lora_down_btn.clicked.connect(lambda: self._on_lora_reorder(+1))
        stack_btn_row.addWidget(self.lora_down_btn)
        stack_btn_row.addWidget(QLabel("Weight"))
        self.lora_weight_spin = QDoubleSpinBox()
        self.lora_weight_spin.setRange(-2.0, 2.0)
        self.lora_weight_spin.setSingleStep(0.05)
        self.lora_weight_spin.setValue(1.0)
        self.lora_weight_spin.valueChanged.connect(self._on_lora_weight_change)
        stack_btn_row.addWidget(self.lora_weight_spin)
        self.lora_remove_btn = QPushButton("Remove")
        self.lora_remove_btn.setObjectName("CancelButton")
        self.lora_remove_btn.clicked.connect(self._on_lora_remove)
        stack_btn_row.addWidget(self.lora_remove_btn)
        self.lora_clear_btn = QPushButton("Clear All")
        self.lora_clear_btn.clicked.connect(self._on_lora_clear)
        stack_btn_row.addWidget(self.lora_clear_btn)
        lp.addLayout(stack_btn_row)

        lay.addWidget(self.lora_panel)

        # Start-frame panel (shown only for I2V models).
        lay.addSpacing(6)
        self.start_frame_panel = QFrame()
        self.start_frame_panel.setObjectName("Panel")
        sf_lay = QVBoxLayout(self.start_frame_panel)
        sf_lay.setContentsMargins(8, 8, 8, 8)
        sf_lay.setSpacing(6)
        sf_lay.addWidget(_section_label("Start Frame (I2V)"))

        self.start_frame_preview = _PreviewLabel()
        self.start_frame_preview.setMinimumHeight(160)
        sf_lay.addWidget(self.start_frame_preview)

        self.start_frame_path_label = QLabel("(none — drop or paste an image, or browse)")
        self.start_frame_path_label.setWordWrap(True)
        self.start_frame_path_label.setStyleSheet("color:#9aa3c8;font-size:11px;")
        sf_lay.addWidget(self.start_frame_path_label)

        sf_btn_row = QHBoxLayout()
        self.start_frame_browse_btn = QPushButton("Browse…")
        self.start_frame_browse_btn.clicked.connect(self._on_start_frame_pick)
        sf_btn_row.addWidget(self.start_frame_browse_btn)
        self.start_frame_from_flux_btn = QPushButton("From Flux Gallery…")
        self.start_frame_from_flux_btn.clicked.connect(self._on_start_frame_from_flux)
        sf_btn_row.addWidget(self.start_frame_from_flux_btn)
        self.start_frame_clear_btn = QPushButton("Clear")
        self.start_frame_clear_btn.clicked.connect(self._on_start_frame_clear)
        sf_btn_row.addWidget(self.start_frame_clear_btn)
        sf_lay.addLayout(sf_btn_row)

        lay.addWidget(self.start_frame_panel)
        lay.addStretch(1)

        # Wire selection -> weight spinbox sync.
        self.lora_list.currentItemChanged.connect(self._on_lora_selection_change)
        return w

    def _populate_model_combo(self) -> None:
        """(Re)fill the model dropdown from _MODEL_ENTRIES."""
        prev = self.model_combo.currentData() if self.model_combo.count() else None
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for name in _MODEL_ENTRIES:
            self.model_combo.addItem(model_display_label(name), name)
        if prev:
            idx = self.model_combo.findData(prev)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    def _build_settings_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # Frames
        lay.addWidget(_section_label("Number of frames"))
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(17, 161)
        self.frames_spin.setSingleStep(4)   # Wan needs (4k+1)
        self.frames_spin.setValue(81)
        self.frames_spin.valueChanged.connect(self._update_duration_label)
        lay.addWidget(self.frames_spin)

        # FPS
        lay.addWidget(_section_label("FPS"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(8, 30)
        self.fps_spin.setValue(15)
        self.fps_spin.valueChanged.connect(self._update_duration_label)
        lay.addWidget(self.fps_spin)

        self.duration_label = QLabel("")
        self.duration_label.setStyleSheet("color:#9aa3c8;font-size:11px;")
        lay.addWidget(self.duration_label)

        # Steps
        lay.addWidget(_section_label("Inference steps"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 100)
        self.steps_spin.setValue(50)
        lay.addWidget(self.steps_spin)

        # Guidance
        lay.addWidget(_section_label("Guidance scale"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 15.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(5.0)
        lay.addWidget(self.guidance_spin)

        # Seed
        lay.addWidget(_section_label("Seed"))
        seed_row = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)
        seed_row.addWidget(self.seed_spin, 1)
        self.randomize_seed_check = QCheckBox("Randomize")
        self.randomize_seed_check.toggled.connect(self._on_randomize_toggled)
        seed_row.addWidget(self.randomize_seed_check)
        lay.addLayout(seed_row)

        lay.addStretch(1)
        self._update_duration_label()
        return w

    def _build_generate_main_area(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Prompts
        prompt_panel = QFrame()
        prompt_panel.setObjectName("Panel")
        plp = QVBoxLayout(prompt_panel)
        plp.setSpacing(6)
        plp.addWidget(_section_label("Prompt"))
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Describe the scene and motion you want Wan to render…"
        )
        self.prompt_edit.setMinimumHeight(70)
        self.prompt_edit.setMaximumHeight(120)
        plp.addWidget(self.prompt_edit)

        plp.addWidget(_section_label("Negative prompt"))
        self.neg_edit = QPlainTextEdit()
        self.neg_edit.setPlainText(DEFAULT_NEGATIVE)
        self.neg_edit.setMinimumHeight(50)
        self.neg_edit.setMaximumHeight(80)
        plp.addWidget(self.neg_edit)
        lay.addWidget(prompt_panel)

        # Preview
        preview_panel = QFrame()
        preview_panel.setObjectName("Panel")
        pp = QVBoxLayout(preview_panel)
        pp.setContentsMargins(8, 8, 8, 8)
        self.video_player = VideoPlayerWidget()
        pp.addWidget(self.video_player, 1)
        lay.addWidget(preview_panel, 1)

        # Bottom controls: progress + buttons + seed display
        bottom = QFrame()
        bottom.setObjectName("Panel")
        bl = QVBoxLayout(bottom)
        bl.setSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("idle")
        bl.addWidget(self.progress_bar)

        info_row = QHBoxLayout()
        self.timer_label = QLabel("")
        self.timer_label.setStyleSheet("color:#c3cbf5;font-weight:600;")
        info_row.addWidget(self.timer_label, 1)
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet("color:#9aa3c8;")
        info_row.addWidget(self.eta_label)
        bl.addLayout(info_row)

        btn_row = QHBoxLayout()
        self.guess_btn = QPushButton("Guess time")
        self.guess_btn.clicked.connect(self._on_guess_time)
        btn_row.addWidget(self.guess_btn)
        self.open_folder_btn = QPushButton("Open Videos folder")
        self.open_folder_btn.clicked.connect(self._open_videos_folder)
        btn_row.addWidget(self.open_folder_btn)
        btn_row.addStretch(1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("CancelButton")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        btn_row.addWidget(self.cancel_btn)
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setObjectName("GenerateButton")
        self.generate_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self.generate_btn)
        bl.addLayout(btn_row)

        # Log box (collapsible — just a small read-only area)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(140)
        self.log_box.setStyleSheet(
            "background:#0a0d18;color:#aeb6d8;font-family:Consolas,monospace;"
            "font-size:11px;border:1px solid #2c3454;border-radius:8px;"
        )
        bl.addWidget(self.log_box)

        lay.addWidget(bottom)

        # Apply initial model-dependent state
        self._on_model_change()
        return w

    def _build_gallery_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        sub = QTabWidget()
        sub.addTab(self._build_gallery_folder(VIDEOS_DIR, kind="video"), "Outputs")
        sub.addTab(self._build_gallery_folder(VIDEO_REF_FRAMES_DIR, kind="image"),
                   "Reference Frames")
        sub.addTab(self._build_gallery_folder(VIDEO_BESTOF_DIR, kind="video"), "Best Of")
        lay.addWidget(sub)
        return w

    def _build_gallery_folder(self, folder: Path, kind: str) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # Header row: refresh + open folder
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(f"Folder: {folder}"))
        hdr.addStretch(1)
        open_btn = QPushButton("Open in Explorer")
        open_btn.clicked.connect(lambda _=None, p=folder: _open_in_explorer(p))
        hdr.addWidget(open_btn)
        refresh_btn = QPushButton("Refresh")
        hdr.addWidget(refresh_btn)
        outer.addLayout(hdr)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # Left: thumbnails
        thumbs = QListWidget()
        thumbs.setViewMode(QListWidget.ViewMode.IconMode)
        thumbs.setIconSize(QSize(160, 90 if kind == "video" else 160))
        thumbs.setGridSize(QSize(180, 130 if kind == "video" else 200))
        thumbs.setResizeMode(QListWidget.ResizeMode.Adjust)
        thumbs.setMovement(QListWidget.Movement.Static)
        thumbs.setSpacing(6)
        thumbs.setMinimumWidth(540)
        splitter.addWidget(thumbs)

        # Middle: preview
        mid = QWidget()
        mid_lay = QVBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        if kind == "video":
            preview_video = VideoPlayerWidget()
            mid_lay.addWidget(preview_video)
            preview_image = None
        else:
            preview_image = _PreviewLabel()
            mid_lay.addWidget(preview_image, 1)
            preview_video = None
        splitter.addWidget(mid)

        # Right: metadata + actions
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(6)
        meta_box = QTextEdit()
        meta_box.setReadOnly(True)
        meta_box.setStyleSheet(
            "background:#0a0d18;color:#aeb6d8;font-family:Consolas,monospace;"
            "font-size:11px;border:1px solid #2c3454;border-radius:8px;"
        )
        r_lay.addWidget(meta_box, 1)

        copy_btn = QPushButton("Copy to Generation")
        as_start_btn = QPushButton("Use as Start Frame")
        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("CancelButton")
        for b in (copy_btn, as_start_btn, delete_btn):
            r_lay.addWidget(b)
        splitter.addWidget(right)

        splitter.setSizes([540, 600, 320])

        state: Dict[str, Any] = {"path": None, "meta": None}

        def populate() -> None:
            thumbs.clear()
            if not folder.exists():
                return
            exts = VIDEO_EXTS if kind == "video" else IMAGE_EXTS
            files = sorted(
                (p for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in exts),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for p in files[:300]:
                if kind == "video":
                    thumb = p.with_suffix(".thumb.jpg")
                    if not thumb.exists():
                        extract_thumbnail(p, thumb)
                    icon = QIcon(str(thumb)) if thumb.exists() else QIcon()
                else:
                    icon = QIcon(str(p))
                item = QListWidgetItem(icon, p.name)
                item.setData(Qt.ItemDataRole.UserRole, str(p))
                thumbs.addItem(item)

        def on_select():
            it = thumbs.currentItem()
            if not it:
                state["path"] = None
                state["meta"] = None
                meta_box.clear()
                if preview_video:
                    preview_video.set_source(None)
                if preview_image:
                    preview_image.set_image(None)
                return
            path = it.data(Qt.ItemDataRole.UserRole)
            state["path"] = path
            if preview_video:
                preview_video.set_source(path, autoplay=False)
            if preview_image:
                preview_image.set_image(path)
            if kind == "video":
                meta = read_sidecar(Path(path)) or {}
                state["meta"] = meta
                meta_box.setPlainText(
                    json.dumps(meta, indent=2, ensure_ascii=False) if meta
                    else "(no sidecar found)"
                )
            else:
                state["meta"] = None
                meta_box.setPlainText(f"file: {path}\n(no sidecar)")

        def on_copy():
            meta = state.get("meta") or {}
            if not meta:
                QMessageBox.information(self, "No metadata",
                                        "This file has no sidecar to copy.")
                return
            self._apply_meta_to_generation(meta)

        def on_use_as_start():
            p = state.get("path")
            if not p:
                return
            if kind == "video":
                thumb = Path(p).with_suffix(".thumb.jpg")
                if not thumb.exists():
                    extract_thumbnail(Path(p), thumb)
                if thumb.exists():
                    self._set_start_frame(str(thumb))
            else:
                self._set_start_frame(p)
            # Auto-switch to I2V if a T2V is selected.
            current = self.model_combo.currentData()
            cur_entry = get_entry(current) if current else None
            if cur_entry and cur_entry["pipeline_kind"] == "t2v":
                for i in range(self.model_combo.count()):
                    name = self.model_combo.itemData(i)
                    e = get_entry(name) if name else None
                    if e and e["pipeline_kind"] == "i2v" and model_exists(name):
                        self.model_combo.setCurrentIndex(i)
                        break
            # Jump back to Generate tab.
            central = self.centralWidget()
            if isinstance(central, QTabWidget):
                central.setCurrentIndex(0)

        def on_delete():
            p = state.get("path")
            if not p:
                return
            if QMessageBox.question(
                self, "Delete file",
                f"Delete this file?\n{p}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return
            try:
                Path(p).unlink(missing_ok=True)
                sidecar = Path(p).with_suffix(".json")
                sidecar.unlink(missing_ok=True)
                thumb = Path(p).with_suffix(".thumb.jpg")
                thumb.unlink(missing_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "Delete failed", str(e))
                return
            populate()

        thumbs.currentItemChanged.connect(lambda *_: on_select())
        refresh_btn.clicked.connect(populate)
        copy_btn.clicked.connect(on_copy)
        as_start_btn.clicked.connect(on_use_as_start)
        delete_btn.clicked.connect(on_delete)

        populate()
        return w

    # ------------------------------------------------------------------
    # Theme — verbatim QSS from flux_image_generator.py (same palette)
    # ------------------------------------------------------------------

    def apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #171b2d;
                color: #e8ecff;
                font-family: Segoe UI, sans-serif;
                font-size: 13px;
            }
            QFrame#Panel {
                background: #1d2238;
                border: 1px solid #2c3454;
                border-radius: 10px;
                padding: 8px;
            }
            QLabel#SectionTitle {
                font-size: 13px; font-weight: 700; color: #c3cbf5; padding: 2px 0;
            }
            QLabel#PreviewFrame {
                border: 1px solid #3a4265;
                border-radius: 10px;
                padding: 6px;
                background: #0f1322;
            }
            QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #0f1322; color: #eef2ff;
                border: 1px solid #3a4265; border-radius: 8px; padding: 7px;
                selection-background-color: #4e67d8;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 28px; }
            QTabWidget#LeftTabs { border: none; }
            QTabWidget#LeftTabs::pane {
                background: #1d2238; border: 1px solid #2c3454;
                border-top: none; border-radius: 0 0 10px 10px;
            }
            QTabBar::tab {
                background: #171b2d; color: #8892b3;
                border: 1px solid #2c3454; border-bottom: none;
                padding: 8px 18px; margin-right: 2px;
                border-radius: 8px 8px 0 0; font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #1d2238; color: #e8ecff;
                border-bottom: 2px solid #6241a6;
            }
            QTabBar::tab:hover:!selected { background: #1a1f35; color: #c3cbf5; }
            QPushButton {
                background: #26335f; color: #f2f4ff;
                border: 1px solid #5264a2; border-radius: 8px;
                padding: 7px 12px; font-weight: 600;
            }
            QPushButton:hover { background: #32427a; }
            QPushButton:disabled {
                background: #1a1e2f; color: #707894; border: 1px solid #2b3045;
            }
            QPushButton#GenerateButton {
                background: #4d2f88; border: 1px solid #9a79df; font-size: 14px;
            }
            QPushButton#GenerateButton:hover { background: #6241a6; }
            QPushButton#CancelButton {
                background: #6b2a2a; border: 1px solid #a84545;
            }
            QPushButton#CancelButton:hover { background: #853535; }
            QProgressBar {
                background: #0f1322; border: 1px solid #3a4265;
                border-radius: 8px; text-align: center;
                color: #e8ecff; font-weight: 600; min-height: 22px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #4d2f88, stop:1 #6241a6);
                border-radius: 7px;
            }
            QStatusBar { background: #101321; color: #aeb6d8; }
        """)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def load_saved_settings(self) -> None:
        s = load_settings()
        model = s.get("model")
        if isinstance(model, str):
            idx = self.model_combo.findData(model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        # Resolution restoration after model dropdown set itself up:
        res_label = s.get("resolution")
        if isinstance(res_label, str):
            idx = self.resolution_combo.findText(res_label)
            if idx >= 0:
                self.resolution_combo.setCurrentIndex(idx)
        if isinstance(s.get("num_frames"), int):
            self.frames_spin.setValue(s["num_frames"])
        if isinstance(s.get("fps"), int):
            self.fps_spin.setValue(s["fps"])
        if isinstance(s.get("num_inference_steps"), int):
            self.steps_spin.setValue(s["num_inference_steps"])
        if isinstance(s.get("guidance_scale"), (int, float)):
            self.guidance_spin.setValue(float(s["guidance_scale"]))
        if isinstance(s.get("seed"), int):
            self.seed_spin.setValue(s["seed"])
        if isinstance(s.get("randomize_seed"), bool):
            self.randomize_seed_check.setChecked(s["randomize_seed"])
        if isinstance(s.get("prompt"), str):
            self.prompt_edit.setPlainText(s["prompt"])
        if isinstance(s.get("negative_prompt"), str):
            self.neg_edit.setPlainText(s["negative_prompt"])
        if isinstance(s.get("start_frame"), str) and Path(s["start_frame"]).exists():
            self._set_start_frame(s["start_frame"])
        # LoRA stack restore — keep only entries whose file still exists.
        raw_stack = s.get("lora_stack")
        if isinstance(raw_stack, list):
            restored: List[Dict[str, Any]] = []
            for item in raw_stack:
                if not isinstance(item, dict):
                    continue
                file = item.get("file")
                weight = item.get("weight", 1.0)
                arch = item.get("arch")
                if not isinstance(file, str) or not Path(file).exists():
                    continue
                if arch not in ("1.3B", "14B"):
                    # Try to infer from parent folder name.
                    parent = Path(file).parent.name
                    arch = parent if parent in ("1.3B", "14B") else None
                if arch is None:
                    continue
                restored.append({
                    "file": file,
                    "weight": float(weight) if isinstance(weight, (int, float)) else 1.0,
                    "arch": arch,
                })
            self._lora_stack = restored
            self._rebuild_lora_list_widget()
        self._update_duration_label()

    def save_current_settings(self) -> None:
        s = {
            "model": self.model_combo.currentData(),
            "resolution": self.resolution_combo.currentText(),
            "num_frames": self.frames_spin.value(),
            "fps": self.fps_spin.value(),
            "num_inference_steps": self.steps_spin.value(),
            "guidance_scale": self.guidance_spin.value(),
            "seed": self.seed_spin.value(),
            "randomize_seed": self.randomize_seed_check.isChecked(),
            "prompt": self.prompt_edit.toPlainText(),
            "negative_prompt": self.neg_edit.toPlainText(),
            "start_frame": self._start_frame_path,
            "lora_stack": list(self._lora_stack),
        }
        save_settings(s)

    def closeEvent(self, ev) -> None:
        self.save_current_settings()
        try:
            self.video_player.stop()
        except Exception:
            pass
        super().closeEvent(ev)

    # ------------------------------------------------------------------
    # Model / settings handlers
    # ------------------------------------------------------------------

    def _on_model_change(self) -> None:
        name = self.model_combo.currentData()
        entry = get_entry(name) if name else None
        if not entry:
            return
        # Resolution list — preserve current selection if still valid.
        prev_label = self.resolution_combo.currentText()
        self.resolution_combo.clear()
        for w, h, label in entry["resolutions"]:
            self.resolution_combo.addItem(label, (w, h))
        idx = self.resolution_combo.findText(prev_label)
        if idx >= 0:
            self.resolution_combo.setCurrentIndex(idx)

        if not model_exists(name):
            target = entry["checkpoint_path"] or entry["base_path"]
            self.model_status_label.setText(
                f"⚠ Not ready: {target}\n"
                f"Download/finish the model before generating with this variant."
            )
            if hasattr(self, "generate_btn"):
                self.generate_btn.setEnabled(False)
        else:
            target = entry["checkpoint_path"] or entry["base_path"]
            tag = "[CKP] " if entry["kind"] == "checkpoint" else ""
            self.model_status_label.setText(f"OK: {tag}{target}")
            if hasattr(self, "generate_btn"):
                self.generate_btn.setEnabled(True)

        # Show/hide start-frame panel
        if hasattr(self, "start_frame_panel"):
            self.start_frame_panel.setVisible(entry["pipeline_kind"] == "i2v")
        # Refresh the LoRA dropdown for the new arch and drop any incompatible
        # LoRAs from the active stack.
        if hasattr(self, "lora_add_combo"):
            self._refresh_lora_dropdown(entry["arch"])
            self._filter_lora_stack_to_arch(entry["arch"])
        self._update_duration_label()

    # ------------------------------------------------------------------
    # Refresh + LoRA stack management
    # ------------------------------------------------------------------

    def _refresh_models(self) -> None:
        rebuild_model_registry()
        self._populate_model_combo()
        # Refresh LoRA dropdown for the current selection's arch.
        name = self.model_combo.currentData()
        entry = get_entry(name) if name else None
        if entry:
            self._refresh_lora_dropdown(entry["arch"])
        sb = self.statusBar()
        if sb is not None:
            sb.showMessage(
                f"Rescanned: {len(_MODEL_ENTRIES)} model entries, "
                f"checkpoints="
                f"{sum(1 for e in _MODEL_ENTRIES.values() if e['kind']=='checkpoint')}.",
                5000,
            )

    def _refresh_lora_dropdown(self, arch: str) -> None:
        self.lora_add_combo.clear()
        for p in scan_loras_for_arch(arch):
            icon = QIcon(str(preview_for_file(p))) if preview_for_file(p) else QIcon()
            self.lora_add_combo.addItem(icon, p.name, str(p))
        if self.lora_add_combo.count() == 0:
            self.lora_add_combo.addItem(
                f"(no LoRAs in {WAN_LORAS_DIR / arch})", None
            )

    def _filter_lora_stack_to_arch(self, arch: str) -> None:
        """Drop stack entries whose arch no longer matches the selected model."""
        dropped = [it for it in self._lora_stack if it.get("arch") != arch]
        if dropped:
            self._lora_stack = [it for it in self._lora_stack
                                if it.get("arch") == arch]
            self._log(
                f"Dropped {len(dropped)} LoRA(s) incompatible with arch {arch}."
            )
        self._rebuild_lora_list_widget()

    def _on_lora_add(self) -> None:
        file = self.lora_add_combo.currentData()
        if not file:
            return
        name = self.model_combo.currentData()
        entry = get_entry(name) if name else None
        if not entry:
            return
        arch = entry["arch"]
        # Avoid duplicates.
        if any(it.get("file") == file for it in self._lora_stack):
            self._log(f"LoRA already in stack: {Path(file).name}")
            return
        self._lora_stack.append({"file": file, "weight": 1.0, "arch": arch})
        self._rebuild_lora_list_widget()
        self.lora_list.setCurrentRow(len(self._lora_stack) - 1)

    def _on_lora_remove(self) -> None:
        row = self.lora_list.currentRow()
        if 0 <= row < len(self._lora_stack):
            self._lora_stack.pop(row)
            self._rebuild_lora_list_widget()

    def _on_lora_clear(self) -> None:
        self._lora_stack = []
        self._rebuild_lora_list_widget()

    def _on_lora_reorder(self, delta: int) -> None:
        row = self.lora_list.currentRow()
        new_row = row + delta
        if row < 0 or new_row < 0 or new_row >= len(self._lora_stack):
            return
        self._lora_stack[row], self._lora_stack[new_row] = \
            self._lora_stack[new_row], self._lora_stack[row]
        self._rebuild_lora_list_widget()
        self.lora_list.setCurrentRow(new_row)

    def _on_lora_weight_change(self, value: float) -> None:
        row = self.lora_list.currentRow()
        if 0 <= row < len(self._lora_stack):
            self._lora_stack[row]["weight"] = float(value)
            # Refresh just the label without rebuilding the whole list.
            self._refresh_lora_row_label(row)

    def _on_lora_selection_change(self, *_) -> None:
        row = self.lora_list.currentRow()
        if 0 <= row < len(self._lora_stack):
            self.lora_weight_spin.blockSignals(True)
            self.lora_weight_spin.setValue(
                float(self._lora_stack[row].get("weight", 1.0))
            )
            self.lora_weight_spin.blockSignals(False)

    def _rebuild_lora_list_widget(self) -> None:
        self.lora_list.clear()
        for item in self._lora_stack:
            file = Path(item["file"])
            label = f"{file.name}   @ {float(item['weight']):.2f}   [{item['arch']}]"
            preview = preview_for_file(file)
            li = QListWidgetItem(QIcon(str(preview)) if preview else QIcon(), label)
            li.setData(Qt.ItemDataRole.UserRole, item["file"])
            self.lora_list.addItem(li)

    def _refresh_lora_row_label(self, row: int) -> None:
        it = self.lora_list.item(row)
        if not it or row >= len(self._lora_stack):
            return
        item = self._lora_stack[row]
        file = Path(item["file"])
        it.setText(
            f"{file.name}   @ {float(item['weight']):.2f}   [{item['arch']}]"
        )

    def _on_randomize_toggled(self, checked: bool) -> None:
        self.seed_spin.setReadOnly(checked)

    def _update_duration_label(self) -> None:
        try:
            frames = self.frames_spin.value()
            fps = max(1, self.fps_spin.value())
            self.duration_label.setText(
                f"≈ {frames / fps:.2f}s clip (frames must be 4k+1; "
                f"e.g. 17, 33, 49, 65, 81, 97, 113)"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Start frame: browse, paste, drag-drop, clear, from-Flux
    # ------------------------------------------------------------------

    def _on_start_frame_pick(self) -> None:
        start_dir = VIDEO_REF_FRAMES_DIR if VIDEO_REF_FRAMES_DIR.exists() else str(APP_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick start frame", str(start_dir),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if path:
            self._set_start_frame(path)

    def _on_start_frame_from_flux(self) -> None:
        start_dir = FLUX_BESTOF_DIR if FLUX_BESTOF_DIR.exists() else FLUX_OUTPUTS_DIR
        if not start_dir.exists():
            QMessageBox.information(
                self, "Flux Gallery missing",
                "Couldn't find the Flux outputs/best-of folder. Generate some "
                "Flux images first, or use Browse… instead.",
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick start frame from Flux Gallery", str(start_dir),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if path:
            self._set_start_frame(path)

    def _on_start_frame_clear(self) -> None:
        self._set_start_frame(None)

    def _set_start_frame(self, path: Optional[str]) -> None:
        self._start_frame_path = path
        if path:
            self.start_frame_preview.set_image(path)
            self.start_frame_path_label.setText(path)
        else:
            self.start_frame_preview.set_image(None)
            self.start_frame_path_label.setText(
                "(none — drop or paste an image, or browse)"
            )

    # Drag/drop + Ctrl+V paste for start frame.
    def dragEnterEvent(self, ev: QDragEnterEvent) -> None:
        if ev.mimeData().hasUrls():
            for url in ev.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in IMAGE_EXTS:
                    ev.acceptProposedAction()
                    return
        ev.ignore()

    def dropEvent(self, ev: QDropEvent) -> None:
        for url in ev.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() in IMAGE_EXTS:
                self._set_start_frame(p)
                ev.acceptProposedAction()
                return

    def keyPressEvent(self, ev) -> None:
        if ev.matches(QKeySequence.StandardKey.Paste):
            # Only handle paste when the prompt/neg editors don't have focus.
            focus = QApplication.focusWidget()
            if focus not in (self.prompt_edit, self.neg_edit):
                cb = QApplication.clipboard()
                md = cb.mimeData()
                if md.hasImage():
                    img = cb.image()
                    if not img.isNull():
                        tmp = VIDEO_REF_FRAMES_DIR / f"_clipboard_{int(time.time())}.png"
                        VIDEO_REF_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
                        if img.save(str(tmp), "PNG"):
                            self._set_start_frame(str(tmp))
                            return
                if md.hasUrls():
                    for url in md.urls():
                        p = url.toLocalFile()
                        if Path(p).suffix.lower() in IMAGE_EXTS:
                            self._set_start_frame(p)
                            return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _on_guess_time(self) -> None:
        cfg = self._collect_config(for_run=False)
        if not cfg:
            return
        est = estimate_total_seconds(
            cfg["model"], cfg["width"], cfg["height"],
            cfg["num_frames"], cfg["num_inference_steps"],
        )
        self.eta_label.setText(f"Estimate: {format_eta(est)}")

    def _collect_config(self, for_run: bool) -> Optional[Dict[str, Any]]:
        model = self.model_combo.currentData()
        if not model:
            QMessageBox.warning(self, "No model", "Pick a model first.")
            return None
        entry = get_entry(model)
        if not entry:
            QMessageBox.warning(self, "Unknown model", model)
            return None
        if for_run and not model_exists(model):
            target = entry["checkpoint_path"] or entry["base_path"]
            QMessageBox.warning(
                self, "Model missing or incomplete",
                f"Not ready on disk:\n{target}",
            )
            return None

        res = self.resolution_combo.currentData()
        if not res:
            QMessageBox.warning(self, "No resolution", "Pick a resolution.")
            return None
        width, height = res

        frames = self.frames_spin.value()
        if (frames - 1) % 4 != 0:
            if for_run:
                QMessageBox.warning(
                    self, "Frame count",
                    f"Wan needs (4k+1) frames. {frames} is not valid; "
                    f"try {((frames - 1) // 4) * 4 + 1} or "
                    f"{((frames - 1) // 4 + 1) * 4 + 1}.",
                )
                return None

        seed = self.seed_spin.value()
        if for_run and self.randomize_seed_check.isChecked():
            seed = random.randint(0, 2_147_483_647)
            self.seed_spin.setValue(seed)

        start_frame = self._start_frame_path
        if entry["pipeline_kind"] == "i2v":
            if not start_frame:
                if for_run:
                    QMessageBox.warning(
                        self, "Start frame required",
                        "I2V models need a start frame. Browse, paste, drop, "
                        "or pick one from the Flux Gallery.",
                    )
                    return None
            if start_frame and not Path(start_frame).exists():
                if for_run:
                    QMessageBox.warning(
                        self, "Start frame missing",
                        f"Start frame file no longer exists: {start_frame}",
                    )
                    return None
        else:
            start_frame = None  # T2V ignores it

        prompt = self.prompt_edit.toPlainText().strip()
        if for_run and not prompt:
            QMessageBox.warning(self, "Empty prompt",
                                "Write something for the prompt first.")
            return None

        # LoRA stack: filter to current arch, validate files exist.
        lora_stack = [
            dict(item) for item in self._lora_stack
            if item.get("arch") == entry["arch"]
        ]
        if for_run:
            valid: List[Dict[str, Any]] = []
            for item in lora_stack:
                p = Path(item.get("file", ""))
                if p.exists():
                    valid.append(item)
                else:
                    self._log(f"LoRA file missing — skipping: {p}")
            lora_stack = valid

        return {
            "model": model,
            "arch": entry["arch"],
            "width": width, "height": height,
            "num_frames": frames,
            "fps": self.fps_spin.value(),
            "num_inference_steps": self.steps_spin.value(),
            "guidance_scale": self.guidance_spin.value(),
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": self.neg_edit.toPlainText().strip(),
            "start_frame": start_frame,
            "lora_stack": lora_stack,
        }

    def _on_generate(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        cfg = self._collect_config(for_run=True)
        if not cfg:
            return
        self.save_current_settings()

        # Estimate for the eta label / countdown target.
        est_total = estimate_total_seconds(
            cfg["model"], cfg["width"], cfg["height"],
            cfg["num_frames"], cfg["num_inference_steps"],
        )
        self._eta_total = est_total
        self.eta_label.setText(f"Estimate: {format_eta(est_total)}")

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("preparing… %p%")
        self.log_box.clear()
        self._log(f"Generating with {cfg['model']} (seed={cfg['seed']})")

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._elapsed_timer.start()
        self._tick_timer.start()

        self.worker = GenerateWorker(cfg)
        self.worker.log.connect(self._log)
        self.worker.progress.connect(self._on_progress)
        self.worker.video_done.connect(self._on_video_done)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_cancel(self) -> None:
        if self.worker and self.worker.isRunning():
            self._log("Cancel requested — will stop at the next step.")
            self.worker.request_cancel()
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, step: int, total: int) -> None:
        pct = int(100 * step / max(1, total))
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"step {step}/{total} (%p%)")

    def _on_video_done(self, path: str, seed: int, meta: Dict[str, Any]) -> None:
        self._current_video_path = path
        self._log(f"Video ready: {path}")
        self.video_player.set_source(path, autoplay=True)

    def _on_completed(self, elapsed: float) -> None:
        self._tick_timer.stop()
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(f"done in {format_eta(elapsed)}")
        self.timer_label.setText(f"Elapsed: {format_eta(elapsed)}")
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.worker = None
        QApplication.alert(self, 0)

    def _on_failed(self, err: str) -> None:
        self._tick_timer.stop()
        self.progress_bar.setFormat(f"failed: {err}")
        self._log(f"FAILED: {err}")
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.worker = None
        QApplication.alert(self, 0)

    def _update_elapsed_display(self) -> None:
        if not self._elapsed_timer.isValid():
            return
        elapsed = self._elapsed_timer.elapsed() / 1000.0
        eta = getattr(self, "_eta_total", 0.0)
        if eta and elapsed < eta:
            remaining = eta - elapsed
            self.timer_label.setText(
                f"Elapsed: {format_eta(elapsed)}  ·  ~{format_eta(remaining)} left"
            )
        elif eta:
            self.timer_label.setText(
                f"Elapsed: {format_eta(elapsed)}  ·  (est. exceeded)"
            )
        else:
            self.timer_label.setText(f"Elapsed: {format_eta(elapsed)}")

    # ------------------------------------------------------------------
    # Gallery: Copy to Generation
    # ------------------------------------------------------------------

    def _apply_meta_to_generation(self, meta: Dict[str, Any]) -> None:
        model = meta.get("model")
        if isinstance(model, str):
            idx = self.model_combo.findData(model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if isinstance(meta.get("width"), int) and isinstance(meta.get("height"), int):
            target = (meta["width"], meta["height"])
            for i in range(self.resolution_combo.count()):
                if self.resolution_combo.itemData(i) == target:
                    self.resolution_combo.setCurrentIndex(i)
                    break
        if isinstance(meta.get("num_frames"), int):
            self.frames_spin.setValue(meta["num_frames"])
        if isinstance(meta.get("fps"), int):
            self.fps_spin.setValue(meta["fps"])
        if isinstance(meta.get("num_inference_steps"), int):
            self.steps_spin.setValue(meta["num_inference_steps"])
        if isinstance(meta.get("guidance_scale"), (int, float)):
            self.guidance_spin.setValue(float(meta["guidance_scale"]))
        if isinstance(meta.get("seed"), int):
            self.seed_spin.setValue(meta["seed"])
            self.randomize_seed_check.setChecked(False)
        if isinstance(meta.get("prompt"), str):
            self.prompt_edit.setPlainText(meta["prompt"])
        if isinstance(meta.get("negative_prompt"), str):
            self.neg_edit.setPlainText(meta["negative_prompt"])
        sf = meta.get("start_frame")
        if isinstance(sf, str) and Path(sf).exists():
            self._set_start_frame(sf)
        # LoRA stack restore — skip files that no longer exist.
        raw_stack = meta.get("lora_stack")
        if isinstance(raw_stack, list):
            cur_name = self.model_combo.currentData()
            cur_entry = get_entry(cur_name) if cur_name else None
            cur_arch = cur_entry["arch"] if cur_entry else None
            restored: List[Dict[str, Any]] = []
            skipped: List[str] = []
            for item in raw_stack:
                if not isinstance(item, dict):
                    continue
                file = item.get("file")
                weight = item.get("weight", 1.0)
                if not isinstance(file, str) or not Path(file).exists():
                    if isinstance(file, str):
                        skipped.append(file)
                    continue
                arch = item.get("arch")
                if arch not in ("1.3B", "14B"):
                    parent = Path(file).parent.name
                    arch = parent if parent in ("1.3B", "14B") else cur_arch
                if arch is None:
                    continue
                restored.append({
                    "file": file,
                    "weight": float(weight) if isinstance(weight, (int, float)) else 1.0,
                    "arch": arch,
                })
            self._lora_stack = restored
            self._rebuild_lora_list_widget()
            if skipped:
                self._log(
                    f"Copy-to-Gen: {len(skipped)} LoRA file(s) missing from disk."
                )
        central = self.centralWidget()
        # Jump to Generate tab.
        if isinstance(central, QTabWidget):
            central.setCurrentIndex(0)

    def _open_videos_folder(self) -> None:
        _open_in_explorer(VIDEOS_DIR)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log_box.appendPlainText(msg)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("SectionTitle")
    return lbl


def _open_in_explorer(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)])
    else:
        subprocess.run(["xdg-open", str(path)])


def qt_exception_hook(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.__excepthook__(exc_type, exc_value, exc_tb)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    sys.excepthook = qt_exception_hook
    app = QApplication(sys.argv)
    app.setApplicationName("FluxVideoGenerator")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
