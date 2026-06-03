#!/usr/bin/env python3
"""
CivitaiBatchResumableDownloader.py

A PyQt6 batch downloader skeleton rebuilt from the working Civitai-aware
Flux LoRA Pair Downloader pattern.

Goals:
- Paste many URLs, one per line.
- Run several downloads at the same time.
- Resolve Civitai / civitai.red model/version/download URLs through the Civitai API first.
- Use a Civitai API token from .env or the GUI token field.
- Resume interrupted downloads with .part files and HTTP Range.
- Retry network interruptions forever by default.
- Never auto-discard/rename a large partial just because a CDN temporarily ignored Range.
- Auto-restore the largest old .ignored_range_backup partial if one exists.
- Refuse to save HTML/JSON/login pages as model files.
- Keep the code clean enough to build your own main downloader on top of it.

Install:
    py -m pip install PyQt6 requests

Run:
    py CivitaiBatchResumableDownloader.py

Portable layout:
    DATA/CivitaiBatchDownloader/.env
    DATA/CivitaiBatchDownloader/downloads
    DATA/CivitaiBatchDownloader/settings.json

It also looks for your old LoRA app token here:
    DATA/FluxLoRADownloader/.env

Supported .env keys:
    CIVITAI_TOKEN="..."
    CIVITAI_API_TOKEN="..."
    CIVITAI_API_KEY="..."
    The_token="..."
    THE_TOKEN="..."
    TOKEN="..."
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from email.message import Message
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlencode, unquote, urlparse, urlunparse

import requests

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# =============================================================================
# App paths / settings
# =============================================================================

APP_NAME = "CivitaiBatchDownloader"
WINDOW_TITLE = "Civitai Batch Resumable Downloader V1.1"

CIVITAI_HOSTS = {
    "civitai.com",
    "www.civitai.com",
    "civitai.red",
    "www.civitai.red",
}

DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 CivitaiBatchDownloader/1.1"
)


def app_root() -> Path:
    """Works both as .py and PyInstaller .exe."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = app_root()
DATA_DIR = ROOT / "DATA" / APP_NAME
DEFAULT_DOWNLOAD_DIR = DATA_DIR / "downloads"
SETTINGS_PATH = DATA_DIR / "settings.json"
ENV_PATH = DATA_DIR / ".env"

# Compatibility path: your existing FluxLoRADownloader token lives here.
LEGACY_FLUX_ENV_PATH = ROOT / "DATA" / "FluxLoRADownloader" / ".env"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    ensure_dirs()
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_settings(settings: dict) -> None:
    ensure_dirs()
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def parse_env_file(path: Path) -> dict:
    """Tiny .env parser so this app does not need python-dotenv."""
    values = {}
    if not path.exists():
        return values

    try:
        text = path.read_text(encoding="utf-8-sig")
    except Exception:
        return values

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        values[key] = value

    return values


def find_civitai_token() -> tuple[str, Optional[Path]]:
    """
    Returns (token, source_path).

    Priority:
    1. DATA/CivitaiBatchDownloader/.env
    2. DATA/FluxLoRADownloader/.env
    """
    keys = [
        "CIVITAI_TOKEN",
        "CIVITAI_API_TOKEN",
        "CIVITAI_API_KEY",
        "The_token",
        "THE_TOKEN",
        "TOKEN",
    ]

    for env_path in [ENV_PATH, LEGACY_FLUX_ENV_PATH]:
        env = parse_env_file(env_path)
        for key in keys:
            value = env.get(key, "").strip()
            if value:
                return value, env_path

    return "", None


# =============================================================================
# General helpers
# =============================================================================

def human_size(num: Optional[int]) -> str:
    if num is None or num < 0:
        return "unknown"

    size = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def sanitize_filename(name: str, fallback: str = "download") -> str:
    name = unquote(name or "").strip().strip("\"'")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(". ")
    return name or fallback


def strip_known_extension(name: str) -> str:
    lowered = name.lower()
    for ext in [
        ".safetensors",
        ".ckpt",
        ".pt",
        ".pth",
        ".bin",
        ".zip",
        ".7z",
        ".rar",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
    ]:
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return name


def filename_from_content_disposition(value: str) -> Optional[str]:
    if not value:
        return None

    msg = Message()
    msg["content-disposition"] = value
    params = msg.get_params(header="content-disposition", unquote=True)

    for key, val in params:
        if key.lower() == "filename" and val:
            return sanitize_filename(str(val))

    match = re.search(r"filename\*\s*=\s*(?:UTF-8''|utf-8'')?([^;]+)", value)
    if match:
        return sanitize_filename(unquote(match.group(1).strip().strip("\"'")))

    return None


def guess_extension_from_url(url: str) -> Optional[str]:
    try:
        suffix = Path(unquote(urlparse(url).path)).suffix.lower()
        if suffix and len(suffix) <= 12:
            return suffix
    except Exception:
        pass
    return None


def extension_from_content_type(content_type: str) -> Optional[str]:
    if not content_type:
        return None

    content_type = content_type.split(";")[0].strip().lower()
    custom = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "application/json": ".json",
        "text/html": ".html",
        "application/octet-stream": None,
        "binary/octet-stream": None,
    }
    if content_type in custom:
        return custom[content_type]

    return mimetypes.guess_extension(content_type)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    for i in range(2, 10000):
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create unique path for: {path}")


def restore_best_partial_backup(part_path: Path, log_func=None) -> None:
    """
    Recover from V1.0's over-protective behavior.

    V1.0 renamed a large .part file to:
        filename.ext.part.ignored_range_backup_<timestamp>
    when a CDN returned HTTP 200 to a Range request.

    That was safe against corruption, but bad UX because the next retry started
    from zero. V1.1 restores the largest backup into the normal .part slot when
    it is bigger than the current .part, so your 11 GB partial becomes resumable
    again automatically.
    """
    try:
        backups = list(part_path.parent.glob(part_path.name + ".ignored_range_backup_*"))
    except Exception:
        return

    if not backups:
        return

    def size_of(path: Path) -> int:
        try:
            return path.stat().st_size
        except Exception:
            return -1

    best = max(backups, key=size_of)
    best_size = size_of(best)
    current_size = size_of(part_path) if part_path.exists() else 0

    if best_size <= current_size:
        return

    try:
        if part_path.exists():
            displaced = part_path.with_name(
                part_path.name + f".smaller_partial_backup_{int(time.time())}"
            )
            part_path.rename(displaced)
            if log_func:
                log_func(
                    f"Found larger backed-up partial. Moved smaller current partial to: {displaced.name}"
                )

        best.rename(part_path)
        if log_func:
            log_func(
                f"Restored backed-up partial for resume: {best.name} -> {part_path.name} "
                f"({human_size(best_size)})"
            )
    except Exception as exc:
        if log_func:
            log_func(f"Could not restore backed-up partial {best.name}: {exc}")


def fallback_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    query = parse_qs(parsed.query)

    # Civitai API download style:
    # /api/download/models/2119045?type=Model&format=SafeTensor&size=full&fp=fp8
    if len(parts) >= 4 and parts[-2].lower() == "models":
        version_id = parts[-1]
        fmt = query.get("format", [""])[0].lower()
        fp = query.get("fp", [""])[0]
        ext = ".safetensors" if "safetensor" in fmt else ".bin"
        suffix = f"_{fp}" if fp else ""
        return sanitize_filename(f"model_{version_id}{suffix}{ext}", "download.bin")

    if parts:
        name = sanitize_filename(parts[-1])
        if "." in name and not name.lower().endswith((".php", ".aspx", ".html")):
            return name

    return f"download_{int(time.time())}.bin"


def add_query_param(url: str, key: str, value: str) -> str:
    if not value:
        return url

    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)

    # Do not overwrite an explicit token already in the pasted URL.
    if key not in query:
        query[key] = [value]

    new_query = urlencode(query, doseq=True)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
        )
    )


def mask_token_in_url(url: str) -> str:
    if not url:
        return url

    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)

    if "token" in query:
        query["token"] = ["***"]

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode(query, doseq=True),
            parsed.fragment,
        )
    )


def is_civitai_url(url: str) -> bool:
    try:
        return urlparse(url).netloc.lower() in CIVITAI_HOSTS
    except Exception:
        return False


def is_probably_transient_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ContentDecodingError,
        ),
    )


def parse_total_size(response: requests.Response, already_have: int) -> Optional[int]:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total_str = content_range.rsplit("/", 1)[-1].strip()
        if total_str.isdigit():
            return int(total_str)

    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        if response.status_code == 206:
            return already_have + int(content_length)
        return int(content_length)

    return None


class PermanentDownloadError(RuntimeError):
    """Do not retry forever; the URL/token/content is bad until the user changes something."""


# =============================================================================
# Civitai resolution
# =============================================================================

def extract_civitai_ids(url: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (model_id, model_version_id) from common Civitai URL shapes."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    model_version_id = None
    model_id = None

    if "modelVersionId" in query:
        try:
            model_version_id = int(query["modelVersionId"][0])
        except Exception:
            pass

    # /api/download/models/12345
    m = re.search(r"/api/download/models/(\d+)", parsed.path)
    if m:
        model_version_id = int(m.group(1))

    # /api/v1/model-versions/12345
    m = re.search(r"/api/v1/model-versions/(\d+)", parsed.path)
    if m:
        model_version_id = int(m.group(1))

    # /models/12345/name-here
    m = re.search(r"/models/(\d+)", parsed.path)
    if m:
        model_id = int(m.group(1))

    return model_id, model_version_id


def source_preferences(source_url: str) -> dict:
    query = parse_qs(urlparse(source_url).query)

    def one(key: str) -> str:
        return str(query.get(key, [""])[0] or "").lower()

    return {
        "type": one("type"),
        "format": one("format"),
        "size": one("size"),
        "fp": one("fp"),
    }


def safe_json_get(url: str, token: str, timeout=(20, 120)) -> dict:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request_url = add_query_param(url, "token", token) if token else url

    response = requests.get(
        request_url,
        headers=headers,
        timeout=timeout,
        allow_redirects=True,
    )

    if response.status_code in {401, 403}:
        raise PermanentDownloadError(
            f"Civitai API auth failed: HTTP {response.status_code}. "
            "Check your Civitai token."
        )

    if response.status_code == 404:
        raise PermanentDownloadError(f"Civitai API returned HTTP 404 for: {url}")

    if response.status_code >= 500 or response.status_code == 429:
        raise requests.exceptions.RetryError(
            f"Civitai API temporary failure: HTTP {response.status_code}"
        )

    if response.status_code >= 400:
        raise PermanentDownloadError(
            f"Civitai API failed: HTTP {response.status_code}\n"
            f"URL: {url}\n"
            f"Response preview: {response.text[:800]}"
        )

    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type.lower():
        raise PermanentDownloadError(
            f"Civitai API did not return JSON.\n"
            f"URL: {url}\n"
            f"Content-Type: {content_type}\n"
            f"Response preview: {response.text[:800]}"
        )

    return response.json()


def choose_best_file(files: list, prefs: Optional[dict] = None) -> Optional[dict]:
    if not files:
        return None

    prefs = prefs or {}
    requested_type = prefs.get("type", "")
    requested_format = prefs.get("format", "")
    requested_size = prefs.get("size", "")
    requested_fp = prefs.get("fp", "")

    def score(f: dict) -> tuple:
        name = str(f.get("name") or "").lower()
        metadata = f.get("metadata") or {}

        file_type = str(f.get("type") or metadata.get("type") or "").lower()
        fmt = str(metadata.get("format") or f.get("format") or "").lower()
        size = str(metadata.get("size") or f.get("size") or "").lower()
        fp = str(metadata.get("fp") or f.get("fp") or "").lower()

        primary = bool(f.get("primary"))
        size_kb = float(f.get("sizeKB") or f.get("sizeKb") or 0)

        safetensor = (
            "safetensor" in fmt
            or "safetensors" in name
            or name.endswith(".safetensors")
        )

        type_match = bool(requested_type and requested_type == file_type)
        format_match = bool(
            requested_format
            and (
                requested_format == fmt
                or ("safetensor" in requested_format and safetensor)
            )
        )
        size_match = bool(requested_size and requested_size == size)
        fp_match = bool(requested_fp and (requested_fp == fp or requested_fp in name))

        # Requested query params should beat "primary" when the user asked for fp8/full/etc.
        return (
            1 if fp_match else 0,
            1 if format_match else 0,
            1 if size_match else 0,
            1 if type_match else 0,
            1 if primary else 0,
            1 if safetensor else 0,
            size_kb,
        )

    return sorted(files, key=score, reverse=True)[0]


def choose_best_image(images: list) -> Optional[dict]:
    if not images:
        return None

    def score(img: dict) -> tuple:
        nsfw = img.get("nsfw")
        safeish = 1 if nsfw in [False, "None", "Soft", "", None] else 0
        width = int(img.get("width") or 0)
        height = int(img.get("height") or 0)
        return (safeish, width * height)

    return sorted(images, key=score, reverse=True)[0]


@dataclass
class ResolvedDownload:
    download_url: str
    filename: str
    details: str
    image_url: str = ""


def resolve_civitai_download(source_url: str, token: str) -> ResolvedDownload:
    model_id, model_version_id = extract_civitai_ids(source_url)

    if not model_id and not model_version_id:
        raise PermanentDownloadError(
            "This looks like a Civitai URL, but I could not find a model ID or modelVersionId in it."
        )

    model_data = None
    version_data = None

    if model_version_id:
        version_api = f"https://civitai.com/api/v1/model-versions/{model_version_id}"
        version_data = safe_json_get(version_api, token)
    elif model_id:
        model_api = f"https://civitai.com/api/v1/models/{model_id}"
        model_data = safe_json_get(model_api, token)
        versions = model_data.get("modelVersions") or []
        if not versions:
            raise PermanentDownloadError("Civitai model has no modelVersions in the API response.")
        version_data = versions[0]
        model_version_id = int(version_data["id"])

    if not version_data:
        raise PermanentDownloadError("Could not resolve Civitai model version data.")

    if model_data is None:
        parent_model = version_data.get("model") or {}
        parent_model_id = parent_model.get("id") or model_id
        if parent_model_id:
            try:
                model_data = safe_json_get(
                    f"https://civitai.com/api/v1/models/{parent_model_id}",
                    token,
                )
            except Exception:
                model_data = None

    files = version_data.get("files") or []
    best_file = choose_best_file(files, source_preferences(source_url))

    download_url = ""
    file_name = ""

    if best_file:
        download_url = str(best_file.get("downloadUrl") or "")
        file_name = str(best_file.get("name") or "")

    if not download_url:
        download_url = str(version_data.get("downloadUrl") or "")

    if not download_url and model_version_id:
        download_url = f"https://civitai.com/api/download/models/{model_version_id}"

    if not download_url:
        raise PermanentDownloadError("Could not find a Civitai download URL for this model version.")

    # Civitai downloads are more reliable with the token query present,
    # while Authorization is also sent in headers.
    if token:
        download_url = add_query_param(download_url, "token", token)

    if file_name:
        filename = sanitize_filename(file_name)
    else:
        model_name = model_data.get("name") if model_data else ""
        version_name = version_data.get("name") or ""
        base = " ".join(x for x in [model_name, version_name] if x).strip()
        if not base:
            base = f"civitai_{model_version_id}"
        filename = sanitize_filename(base)

        if not Path(filename).suffix:
            filename += ".safetensors"

    images = version_data.get("images") or []
    best_image = choose_best_image(images)
    image_url = str(best_image.get("url") or "") if best_image else ""

    details = []
    if model_data and model_data.get("name"):
        details.append(f"Model: {model_data.get('name')}")
    if version_data.get("name"):
        details.append(f"Version: {version_data.get('name')}")
    if model_version_id:
        details.append(f"ModelVersionId: {model_version_id}")
    if file_name:
        details.append(f"File: {file_name}")
    if best_file:
        metadata = best_file.get("metadata") or {}
        if metadata:
            compact_meta = ", ".join(f"{k}={v}" for k, v in metadata.items() if v)
            if compact_meta:
                details.append(f"Metadata: {compact_meta}")
    if images:
        details.append(f"Images found: {len(images)}")

    return ResolvedDownload(
        download_url=download_url,
        filename=filename,
        details="\n".join(details),
        image_url=image_url,
    )


# =============================================================================
# Download job / worker
# =============================================================================

@dataclass
class DownloadJob:
    id: str
    source_url: str
    output_dir: Path
    overwrite: bool
    download_preview_image: bool
    token: str
    filename_hint: str = ""
    status: str = "Queued"
    downloaded: int = 0
    total: Optional[int] = None
    speed: float = 0.0
    final_path: Optional[Path] = None
    worker: Optional["DownloadWorker"] = None
    progress_bar: Optional[QProgressBar] = field(default=None, repr=False)


class DownloadWorker(QThread):
    status_changed = pyqtSignal(str, str)
    progress_changed = pyqtSignal(str, object, object, float)
    filename_changed = pyqtSignal(str, str, str)
    log_message = pyqtSignal(str, str)
    finished_job = pyqtSignal(str, bool, str)

    def __init__(
        self,
        job: DownloadJob,
        retry_delay: int = 60,
        max_retries: int = -1,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.job = job
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def headers(self) -> dict:
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            # Important for Range downloads. We want byte offsets in the real file,
            # not in a transparent compressed transfer.
            "Accept-Encoding": "identity",
        }
        if self.job.token.strip():
            headers["Authorization"] = f"Bearer {self.job.token.strip()}"
        return headers

    def log(self, message: str) -> None:
        self.log_message.emit(self.job.id, message)

    def status(self, message: str) -> None:
        self.status_changed.emit(self.job.id, message)

    def sleep_interruptibly(self, seconds: int) -> None:
        end = time.time() + seconds
        while time.time() < end:
            if self._cancelled:
                return
            time.sleep(0.25)

    def run(self) -> None:
        try:
            self.job.output_dir.mkdir(parents=True, exist_ok=True)

            attempt = 0
            resolved: Optional[ResolvedDownload] = None

            while not self._cancelled:
                try:
                    self.status("Resolving" if is_civitai_url(self.job.source_url) else "Preparing")

                    if is_civitai_url(self.job.source_url):
                        resolved = resolve_civitai_download(self.job.source_url, self.job.token)
                        self.log("Resolved Civitai source:\n" + resolved.details)
                        download_url = resolved.download_url
                        filename = resolved.filename
                    else:
                        download_url = self.job.source_url
                        filename = self.job.filename_hint or fallback_filename_from_url(download_url)

                    target_path = self.job.output_dir / sanitize_filename(filename, "download.bin")

                    if not self.job.overwrite and target_path.exists():
                        # If completed file exists, make a new unique name.
                        # If only .part exists, resume that .part.
                        part_for_existing = target_path.with_name(target_path.name + ".part")
                        if not part_for_existing.exists():
                            target_path = unique_path(target_path)

                    self.filename_changed.emit(self.job.id, target_path.name, str(target_path))

                    final_path = self.download_file_resumable(
                        url=download_url,
                        target_path=target_path,
                        validate_binary=True,
                    )
                    self.job.final_path = final_path

                    if (
                        self.job.download_preview_image
                        and resolved is not None
                        and resolved.image_url
                    ):
                        try:
                            self.status("Downloading preview")
                            image_ext = guess_extension_from_url(resolved.image_url) or ".jpg"
                            image_path = final_path.with_suffix(image_ext)
                            if not self.job.overwrite and image_path.exists():
                                image_path = unique_path(image_path)

                            self.download_file_resumable(
                                url=resolved.image_url,
                                target_path=image_path,
                                validate_binary=False,
                                allow_suffix_change=True,
                                update_main_progress=False,
                            )
                            self.log(f"Saved preview image: {image_path.name}")
                        except Exception as image_exc:
                            self.log(f"Preview image failed, model file is still complete: {image_exc}")

                    self.finished_job.emit(self.job.id, True, f"Done: {final_path}")
                    return

                except PermanentDownloadError as exc:
                    self.finished_job.emit(self.job.id, False, str(exc))
                    return

                except Exception as exc:
                    if self._cancelled:
                        self.finished_job.emit(self.job.id, False, "Paused")
                        return

                    attempt += 1

                    # If it is not a classic connection/timeout error, still retry unless the user stops it.
                    # This keeps the app useful during long network outages/CDN instability.
                    self.log(f"Attempt {attempt} failed: {exc}")

                    if self.max_retries >= 0 and attempt > self.max_retries:
                        self.finished_job.emit(self.job.id, False, f"Retry limit reached: {exc}")
                        return

                    self.status(f"Retrying in {self.retry_delay}s")
                    self.sleep_interruptibly(self.retry_delay)

            self.finished_job.emit(self.job.id, False, "Paused")

        except Exception as fatal:
            self.finished_job.emit(
                self.job.id,
                False,
                f"Fatal worker error: {fatal}\n\n{traceback.format_exc()}",
            )

    def download_file_resumable(
        self,
        url: str,
        target_path: Path,
        validate_binary: bool,
        allow_suffix_change: bool = False,
        update_main_progress: bool = True,
    ) -> Path:
        url = url.strip()
        if not url:
            raise PermanentDownloadError("Missing download URL.")

        target_path = target_path.resolve()
        part_path = target_path.with_name(target_path.name + ".part")

        while not self._cancelled:
            # Recover partials that V1.0 backed up when a CDN temporarily ignored Range.
            restore_best_partial_backup(part_path, self.log)

            already_have = part_path.stat().st_size if part_path.exists() else 0

            headers = self.headers()
            if already_have > 0:
                headers["Range"] = f"bytes={already_have}-"
                self.status(f"Resuming from {human_size(already_have)}")
            else:
                self.status("Connecting")

            with requests.get(
                url,
                stream=True,
                timeout=(20, 180),
                headers=headers,
                allow_redirects=True,
            ) as response:
                status = response.status_code
                content_type = response.headers.get("Content-Type", "")
                content_length = response.headers.get("Content-Length", "")
                content_disposition = response.headers.get("Content-Disposition", "")

                self.log(
                    f"HTTP {status}; type={content_type or 'unknown'}; "
                    f"length={content_length or 'unknown'}; url={mask_token_in_url(response.url)}"
                )

                if status in {401, 403}:
                    preview = self.safe_response_preview(response)
                    raise PermanentDownloadError(
                        f"Auth failed: HTTP {status}. Check your token.\n{preview}"
                    )

                if status == 404:
                    raise PermanentDownloadError(f"HTTP 404. File not found: {mask_token_in_url(url)}")

                if status == 416:
                    # Requested range not satisfiable. Usually the .part is already complete.
                    if part_path.exists() and part_path.stat().st_size > 0:
                        if target_path.exists() and self.job.overwrite:
                            target_path.unlink()
                        part_path.replace(target_path)
                        size = target_path.stat().st_size
                        if update_main_progress:
                            self.progress_changed.emit(self.job.id, size, size, 0.0)
                        return target_path
                    raise PermanentDownloadError("HTTP 416 but no partial file exists.")

                if status >= 500 or status == 429:
                    raise requests.exceptions.RetryError(f"Temporary HTTP {status}")

                if status >= 400:
                    preview = self.safe_response_preview(response)
                    raise PermanentDownloadError(
                        f"Download failed: HTTP {status}\nURL: {mask_token_in_url(url)}\n{preview}"
                    )

                if already_have > 0 and status != 206:
                    # Server ignored Range. Appending would corrupt the file.
                    #
                    # V1.0 renamed the partial and restarted from zero here. That was
                    # safe, but it wasted huge near-complete downloads. V1.1 never
                    # discards or hides the partial automatically. It keeps the .part
                    # in place and retries later, because Civitai/B2 signed URLs can
                    # sometimes refuse Range temporarily and then accept it on a later
                    # signed URL.
                    full_len = int(content_length) if content_length and content_length.isdigit() else -1

                    if full_len > 0 and already_have >= full_len:
                        # The partial is already at least as large as the full file the
                        # server is offering. This can happen after a broken final read.
                        # Validate and finalize instead of starting over.
                        self.log(
                            "Server returned HTTP 200 to a Range request, but the partial "
                            "is already at least the advertised full size. Validating and finalizing partial."
                        )
                        if validate_binary:
                            self.validate_downloaded_file(part_path)
                        if target_path.exists() and self.job.overwrite:
                            target_path.unlink()
                        part_path.replace(target_path)
                        if update_main_progress:
                            final_size = target_path.stat().st_size
                            self.progress_changed.emit(self.job.id, final_size, final_size, 0.0)
                        self.log(f"Saved: {target_path.name} ({human_size(target_path.stat().st_size)})")
                        return target_path

                    raise requests.exceptions.RetryError(
                        "Server returned HTTP 200 to a Range resume request. "
                        f"Keeping existing partial ({human_size(already_have)}) and retrying later; "
                        "not restarting from zero."
                    )

                if validate_binary and self.response_is_text_error(content_type):
                    preview = self.safe_response_preview(response)
                    raise PermanentDownloadError(
                        "Server returned HTML/JSON/text instead of a binary file. "
                        "This is usually an auth page, API error, or wrong URL.\n"
                        f"Content-Type: {content_type}\n{preview}"
                    )

                # For direct non-Civitai URLs, use server filename if it provides one.
                cd_name = filename_from_content_disposition(content_disposition)
                if cd_name and cd_name != target_path.name:
                    new_target = target_path.with_name(cd_name)
                    if not self.job.overwrite and new_target.exists():
                        new_part = new_target.with_name(new_target.name + ".part")
                        if not new_part.exists():
                            new_target = unique_path(new_target)

                    # If target name changes, restart this method with the new path so resume checks are correct.
                    self.filename_changed.emit(self.job.id, new_target.name, str(new_target))
                    target_path = new_target
                    part_path = target_path.with_name(target_path.name + ".part")
                    if part_path.exists() and "Range" not in headers:
                        continue

                if allow_suffix_change:
                    detected_ext = extension_from_content_type(content_type)
                    if detected_ext and detected_ext not in {".html", ".json"}:
                        if target_path.suffix.lower() != detected_ext:
                            target_path = target_path.with_suffix(detected_ext)
                            part_path = target_path.with_name(target_path.name + ".part")

                total_size = parse_total_size(response, already_have)
                downloaded = already_have
                started = time.time()

                self.status("Downloading")
                if update_main_progress:
                    self.progress_changed.emit(self.job.id, downloaded, total_size or -1, 0.0)

                mode = "ab" if already_have > 0 else "wb"

                with open(part_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if self._cancelled:
                            self.status("Paused")
                            raise PermanentDownloadError("Paused")

                        if not chunk:
                            continue

                        f.write(chunk)
                        downloaded += len(chunk)

                        if update_main_progress:
                            elapsed = max(time.time() - started, 0.001)
                            speed = max((downloaded - already_have) / elapsed, 0.0)
                            self.progress_changed.emit(
                                self.job.id,
                                downloaded,
                                total_size or -1,
                                speed,
                            )

                if total_size is not None and downloaded != total_size:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"Connection ended early: have {human_size(downloaded)}, "
                        f"expected {human_size(total_size)}"
                    )

            if validate_binary:
                self.validate_downloaded_file(part_path)

            if target_path.exists() and self.job.overwrite:
                target_path.unlink()

            part_path.replace(target_path)

            if update_main_progress:
                final_size = target_path.stat().st_size
                self.progress_changed.emit(self.job.id, final_size, final_size, 0.0)

            self.log(f"Saved: {target_path.name} ({human_size(target_path.stat().st_size)})")
            return target_path

        raise PermanentDownloadError("Paused")

    @staticmethod
    def response_is_text_error(content_type: str) -> bool:
        lowered = (content_type or "").lower()
        return (
            "text/html" in lowered
            or "text/plain" in lowered
            or "application/json" in lowered
            or "application/problem+json" in lowered
        )

    @staticmethod
    def safe_response_preview(response: requests.Response, limit: int = 1200) -> str:
        try:
            text = response.text[:limit]
            if text:
                return "Response preview:\n" + text
        except Exception:
            pass
        return ""

    @staticmethod
    def validate_downloaded_file(part_path: Path) -> None:
        if not part_path.exists():
            raise PermanentDownloadError(f"Download temp file is missing: {part_path}")

        size = part_path.stat().st_size

        # Not every useful download is huge, but model files should not be tiny HTML/JSON.
        # Keep this as a content check, not a blind size check.
        try:
            prefix = part_path.read_bytes()[:2048].lower().lstrip()
        except Exception:
            prefix = b""

        if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
            bad = part_path.with_name(part_path.name + f".html_error_{int(time.time())}")
            part_path.rename(bad)
            raise PermanentDownloadError(
                f"Downloaded content was HTML, not the real file. Moved bad file to: {bad.name}"
            )

        if prefix.startswith(b"{") and b"error" in prefix[:512]:
            bad = part_path.with_name(part_path.name + f".json_error_{int(time.time())}")
            part_path.rename(bad)
            raise PermanentDownloadError(
                f"Downloaded content looked like a JSON error, not the real file. Moved bad file to: {bad.name}"
            )

        if size == 0:
            raise PermanentDownloadError("Downloaded file is 0 bytes.")


# =============================================================================
# GUI
# =============================================================================

class MainWindow(QMainWindow):
    COL_FILE = 0
    COL_STATUS = 1
    COL_PROGRESS = 2
    COL_SIZE = 3
    COL_SPEED = 4
    COL_URL = 5
    COL_PATH = 6

    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()

        self.settings = load_settings()
        self.jobs: dict[str, DownloadJob] = {}
        self.row_to_job: dict[int, str] = {}

        self.token, self.token_path = find_civitai_token()

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1420, 820)

        self.build_ui()
        self.build_menu()

        self.queue_timer = QTimer(self)
        self.queue_timer.timeout.connect(self.pump_queue)
        self.queue_timer.start(1000)

    def build_ui(self) -> None:
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root = QVBoxLayout(root_widget)

        intro = QLabel(
            "Paste multiple Civitai/civitai.red links or direct file URLs, one per line. "
            "Civitai links are resolved through the API first, then downloaded with resume support."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.token_status = QLabel()
        self.token_status.setWordWrap(True)
        root.addWidget(self.token_status)

        self.url_input = QPlainTextEdit()
        self.url_input.setPlaceholderText(
            "https://civitai.red/api/download/models/2119045?type=Model&format=SafeTensor&size=full&fp=fp8\n"
            "https://civitai.com/models/12345/model-name?modelVersionId=67890\n"
            "https://example.com/file.zip"
        )
        self.url_input.setMinimumHeight(95)
        root.addWidget(self.url_input)

        grid = QGridLayout()

        self.output_dir = QLineEdit()
        self.output_dir.setText(
            self.settings.get(
                "default_output_dir",
                self.settings.get("output_dir", str(DEFAULT_DOWNLOAD_DIR)),
            )
        )

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_output_dir)

        set_default_btn = QPushButton("Set Default Output Folder")
        set_default_btn.clicked.connect(self.set_default_output_folder)

        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("Optional Civitai token override. Empty = use .env token.")
        if self.token:
            self.token_edit.setText(self.token)

        reload_token_btn = QPushButton("Reload Token")
        reload_token_btn.clicked.connect(self.reload_token)

        open_env_btn = QPushButton("Open DATA Folder")
        open_env_btn.clicked.connect(self.open_data_folder)

        self.concurrent_spin = QSpinBox()
        self.concurrent_spin.setRange(1, 16)
        self.concurrent_spin.setValue(int(self.settings.get("max_simultaneous", 3)))

        self.retry_delay_spin = QSpinBox()
        self.retry_delay_spin.setRange(5, 3600)
        self.retry_delay_spin.setValue(int(self.settings.get("retry_delay", 60)))
        self.retry_delay_spin.setSuffix(" sec")

        self.overwrite_box = QCheckBox("Overwrite completed files")
        self.overwrite_box.setChecked(bool(self.settings.get("overwrite", False)))

        self.preview_box = QCheckBox("Also download Civitai preview image")
        self.preview_box.setChecked(bool(self.settings.get("download_preview_image", False)))

        grid.addWidget(QLabel("Output folder:"), 0, 0)
        grid.addWidget(self.output_dir, 0, 1, 1, 3)
        grid.addWidget(browse_btn, 0, 4)
        grid.addWidget(set_default_btn, 0, 5)

        grid.addWidget(QLabel("Civitai token:"), 1, 0)
        grid.addWidget(self.token_edit, 1, 1, 1, 3)
        grid.addWidget(reload_token_btn, 1, 4)
        grid.addWidget(open_env_btn, 1, 5)

        grid.addWidget(QLabel("Max simultaneous:"), 2, 0)
        grid.addWidget(self.concurrent_spin, 2, 1)
        grid.addWidget(QLabel("Retry delay:"), 2, 2)
        grid.addWidget(self.retry_delay_spin, 2, 3)
        grid.addWidget(self.overwrite_box, 2, 4)
        grid.addWidget(self.preview_box, 2, 5)

        root.addLayout(grid)

        btns = QHBoxLayout()

        paste_btn = QPushButton("Paste Links")
        paste_btn.clicked.connect(self.paste_links)

        add_btn = QPushButton("Add Links to Queue")
        add_btn.clicked.connect(self.add_links_to_queue)

        start_btn = QPushButton("Start / Resume Queue")
        start_btn.clicked.connect(self.start_resume_queue)

        pause_selected_btn = QPushButton("Pause Selected")
        pause_selected_btn.clicked.connect(self.pause_selected)

        pause_all_btn = QPushButton("Pause All")
        pause_all_btn.clicked.connect(self.pause_all)

        remove_selected_btn = QPushButton("Remove Selected")
        remove_selected_btn.clicked.connect(self.remove_selected)

        open_folder_btn = QPushButton("Open Download Folder")
        open_folder_btn.clicked.connect(self.open_output_folder)

        btns.addWidget(paste_btn)
        btns.addWidget(add_btn)
        btns.addWidget(start_btn)
        btns.addWidget(pause_selected_btn)
        btns.addWidget(pause_all_btn)
        btns.addWidget(remove_selected_btn)
        btns.addStretch(1)
        btns.addWidget(open_folder_btn)

        root.addLayout(btns)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["File", "Status", "Progress", "Size", "Speed", "URL", "Final Path"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.COL_FILE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_PROGRESS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_SIZE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_SPEED, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_URL, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_PATH, QHeaderView.ResizeMode.Stretch)

        root.addWidget(self.table, stretch=1)

        root.addWidget(QLabel("Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(170)
        root.addWidget(self.log_box)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.")
        self.refresh_token_label()

    def build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        add_action = QAction("Add Links to Queue", self)
        add_action.triggered.connect(self.add_links_to_queue)
        file_menu.addAction(add_action)

        start_action = QAction("Start / Resume Queue", self)
        start_action.triggered.connect(self.start_resume_queue)
        file_menu.addAction(start_action)

        pause_action = QAction("Pause All", self)
        pause_action.triggered.connect(self.pause_all)
        file_menu.addAction(pause_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # -------------------------------------------------------------------------
    # Settings / token
    # -------------------------------------------------------------------------

    def current_token(self) -> str:
        return self.token_edit.text().strip()

    def refresh_token_label(self) -> None:
        if self.current_token():
            source = str(self.token_path) if self.token_path else "GUI token field"
            self.token_status.setText(f"Civitai token loaded/available. Source: {source}")
        else:
            self.token_status.setText(
                "No Civitai token found. Put it in DATA/CivitaiBatchDownloader/.env "
                "or DATA/FluxLoRADownloader/.env, or paste it into the token box."
            )

    def reload_token(self) -> None:
        self.token, self.token_path = find_civitai_token()
        self.token_edit.setText(self.token)
        self.refresh_token_label()
        self.add_log("APP", "Reloaded token from .env." if self.token else "No token found in .env.")

    def persist_settings(self) -> None:
        self.settings["output_dir"] = self.output_dir.text().strip()
        self.settings["default_output_dir"] = self.output_dir.text().strip()
        self.settings["max_simultaneous"] = self.concurrent_spin.value()
        self.settings["retry_delay"] = self.retry_delay_spin.value()
        self.settings["overwrite"] = self.overwrite_box.isChecked()
        self.settings["download_preview_image"] = self.preview_box.isChecked()
        save_settings(self.settings)

    def set_default_output_folder(self) -> None:
        folder = Path(self.output_dir.text().strip() or DEFAULT_DOWNLOAD_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        self.output_dir.setText(str(folder))
        self.persist_settings()
        self.add_log("APP", f"Default output folder saved: {folder}")

    # -------------------------------------------------------------------------
    # Queue operations
    # -------------------------------------------------------------------------

    def paste_links(self) -> None:
        text = QGuiApplication.clipboard().text().strip()
        if text:
            self.url_input.setPlainText(text)

    def browse_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose download folder",
            self.output_dir.text().strip() or str(DEFAULT_DOWNLOAD_DIR),
        )
        if selected:
            self.output_dir.setText(selected)
            self.persist_settings()

    def add_links_to_queue(self) -> None:
        raw = self.url_input.toPlainText()
        urls = [line.strip() for line in raw.splitlines() if line.strip()]

        if not urls:
            self.statusBar().showMessage("No links to add.")
            return

        output_dir = Path(self.output_dir.text().strip() or DEFAULT_DOWNLOAD_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.persist_settings()
        token = self.current_token()

        existing = {job.source_url for job in self.jobs.values()}
        added = 0

        for url in urls:
            if not url.lower().startswith(("http://", "https://")):
                self.add_log("APP", f"Skipped invalid URL: {url}")
                continue

            if url in existing:
                self.add_log("APP", f"Skipped duplicate already in queue: {url}")
                continue

            if is_civitai_url(url) and not token and "token=" not in url:
                self.add_log("APP", f"Civitai URL added without token. It may fail if login is required: {url}")

            job_id = uuid.uuid4().hex
            hint = fallback_filename_from_url(url)

            job = DownloadJob(
                id=job_id,
                source_url=url,
                output_dir=output_dir,
                overwrite=self.overwrite_box.isChecked(),
                download_preview_image=self.preview_box.isChecked(),
                token=token,
                filename_hint=hint,
                final_path=output_dir / hint,
            )

            self.jobs[job_id] = job
            self.add_job_row(job)
            added += 1

        self.url_input.clear()
        self.statusBar().showMessage(f"Added {added} download(s).")

    def add_job_row(self, job: DownloadJob) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.row_to_job[row] = job.id

        file_item = QTableWidgetItem(job.filename_hint or "Resolving...")
        file_item.setData(Qt.ItemDataRole.UserRole, job.id)

        self.table.setItem(row, self.COL_FILE, file_item)
        self.table.setItem(row, self.COL_STATUS, QTableWidgetItem(job.status))

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setFormat("0%")
        job.progress_bar = bar
        self.table.setCellWidget(row, self.COL_PROGRESS, bar)

        self.table.setItem(row, self.COL_SIZE, QTableWidgetItem("unknown"))
        self.table.setItem(row, self.COL_SPEED, QTableWidgetItem("-"))
        self.table.setItem(row, self.COL_URL, QTableWidgetItem(job.source_url))
        self.table.setItem(row, self.COL_PATH, QTableWidgetItem(str(job.final_path or "")))

    def start_resume_queue(self) -> None:
        self.persist_settings()
        token = self.current_token()
        output_dir = Path(self.output_dir.text().strip() or DEFAULT_DOWNLOAD_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        for job in self.jobs.values():
            # Refresh settings for queued/failed/paused jobs.
            if job.status in {"Queued", "Paused", "Failed", "Needs token/auth"}:
                job.output_dir = output_dir
                job.overwrite = self.overwrite_box.isChecked()
                job.download_preview_image = self.preview_box.isChecked()
                job.token = token
                job.status = "Queued"
                self.update_job_status(job.id, "Queued")

        self.pump_queue()
        self.statusBar().showMessage("Queue started/resumed.")

    def active_count(self) -> int:
        return sum(
            1
            for job in self.jobs.values()
            if job.worker is not None and job.worker.isRunning()
        )

    def pump_queue(self) -> None:
        max_active = self.concurrent_spin.value()

        while self.active_count() < max_active:
            next_job = None
            for job in self.jobs.values():
                if job.status == "Queued" and job.worker is None:
                    next_job = job
                    break

            if next_job is None:
                return

            self.start_job(next_job)

    def start_job(self, job: DownloadJob) -> None:
        worker = DownloadWorker(
            job=job,
            retry_delay=self.retry_delay_spin.value(),
            max_retries=-1,
            chunk_size=DEFAULT_CHUNK_SIZE,
        )

        worker.status_changed.connect(self.update_job_status)
        worker.progress_changed.connect(self.update_job_progress)
        worker.filename_changed.connect(self.update_job_filename)
        worker.log_message.connect(self.add_log)
        worker.finished_job.connect(self.handle_job_finished)

        job.worker = worker
        job.status = "Starting"
        self.update_job_status(job.id, "Starting")
        worker.start()

    def selected_job_ids(self) -> list[str]:
        ids: list[str] = []
        for index in self.table.selectionModel().selectedRows():
            job_id = self.row_to_job.get(index.row())
            if job_id:
                ids.append(job_id)
        return ids

    def pause_selected(self) -> None:
        ids = self.selected_job_ids()
        if not ids:
            self.statusBar().showMessage("No selected jobs.")
            return

        for job_id in ids:
            self.pause_job(job_id)

        self.statusBar().showMessage(f"Pause requested for {len(ids)} job(s).")

    def pause_all(self) -> None:
        for job_id in list(self.jobs.keys()):
            self.pause_job(job_id)

        self.statusBar().showMessage("Pause requested for all jobs.")

    def pause_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return

        if job.worker is not None and job.worker.isRunning():
            job.worker.cancel()
            self.update_job_status(job_id, "Pausing...")
        elif job.status == "Queued":
            job.status = "Paused"
            self.update_job_status(job_id, "Paused")

    def remove_selected(self) -> None:
        ids = set(self.selected_job_ids())
        if not ids:
            self.statusBar().showMessage("No selected jobs.")
            return

        removed = 0

        for row in reversed(range(self.table.rowCount())):
            job_id = self.row_to_job.get(row)
            if job_id not in ids:
                continue

            job = self.jobs.get(job_id)
            if not job:
                continue

            if job.worker is not None and job.worker.isRunning():
                self.add_log(job_id, "Cannot remove while running. Pause first.")
                continue

            self.jobs.pop(job_id, None)
            self.table.removeRow(row)
            removed += 1

        self.rebuild_row_map()
        self.statusBar().showMessage(f"Removed {removed} job(s).")

    # -------------------------------------------------------------------------
    # UI updates
    # -------------------------------------------------------------------------

    def rebuild_row_map(self) -> None:
        self.row_to_job.clear()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_FILE)
            if item:
                job_id = item.data(Qt.ItemDataRole.UserRole)
                if job_id:
                    self.row_to_job[row] = job_id

    def find_row(self, job_id: str) -> Optional[int]:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_FILE)
            if item and item.data(Qt.ItemDataRole.UserRole) == job_id:
                return row
        return None

    def update_job_status(self, job_id: str, status: str) -> None:
        job = self.jobs.get(job_id)
        if job:
            job.status = status

        row = self.find_row(job_id)
        if row is None:
            return

        item = self.table.item(row, self.COL_STATUS)
        if item:
            item.setText(status)
        else:
            self.table.setItem(row, self.COL_STATUS, QTableWidgetItem(status))

    def update_job_filename(self, job_id: str, filename: str, final_path: str) -> None:
        job = self.jobs.get(job_id)
        if job:
            job.filename_hint = filename
            job.final_path = Path(final_path)

        row = self.find_row(job_id)
        if row is None:
            return

        file_item = self.table.item(row, self.COL_FILE)
        if file_item:
            file_item.setText(filename)

        path_item = self.table.item(row, self.COL_PATH)
        if path_item:
            path_item.setText(final_path)

    def update_job_progress(self, job_id: str, downloaded_obj: object, total_obj: object, speed: float) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return

        try:
            downloaded = int(downloaded_obj)
        except Exception:
            downloaded = 0

        try:
            total = int(total_obj)
        except Exception:
            total = -1

        job.downloaded = downloaded
        job.total = total if total >= 0 else None
        job.speed = speed

        row = self.find_row(job_id)
        if row is None:
            return

        if job.progress_bar:
            if total > 0:
                pct = max(0, min(100, int(downloaded * 100 / total)))
                job.progress_bar.setRange(0, 100)
                job.progress_bar.setValue(pct)
                job.progress_bar.setFormat(f"{pct}%")
            else:
                job.progress_bar.setRange(0, 0)
                job.progress_bar.setFormat(human_size(downloaded))

        size_item = self.table.item(row, self.COL_SIZE)
        if size_item:
            size_item.setText(f"{human_size(downloaded)} / {human_size(total if total >= 0 else None)}")

        speed_item = self.table.item(row, self.COL_SPEED)
        if speed_item:
            speed_item.setText(f"{human_size(int(speed))}/s" if speed > 0 else "-")

    def handle_job_finished(self, job_id: str, success: bool, message: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return

        if job.worker is not None:
            job.worker.quit()
            job.worker.wait(100)
            job.worker = None

        if success:
            job.status = "Done"
            self.update_job_status(job_id, "Done")
            self.add_log(job_id, message)
        else:
            if message == "Paused":
                job.status = "Paused"
                self.update_job_status(job_id, "Paused")
                self.add_log(job_id, "Paused. Partial .part file was kept.")
            elif "token" in message.lower() or "auth" in message.lower():
                job.status = "Needs token/auth"
                self.update_job_status(job_id, "Needs token/auth")
                self.add_log(job_id, message)
            else:
                job.status = "Failed"
                self.update_job_status(job_id, "Failed")
                self.add_log(job_id, "FAILED: " + message)

        self.pump_queue()

    def add_log(self, job_id: str, message: str) -> None:
        if job_id == "APP":
            prefix = "APP"
        else:
            job = self.jobs.get(job_id)
            prefix = job.filename_hint if job else job_id[:8]

        timestamp = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{timestamp}] [{prefix}] {message}")

    # -------------------------------------------------------------------------
    # Folder operations / close
    # -------------------------------------------------------------------------

    def open_output_folder(self) -> None:
        folder = Path(self.output_dir.text().strip() or DEFAULT_DOWNLOAD_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        webbrowser.open(folder.as_uri())

    def open_data_folder(self) -> None:
        ensure_dirs()
        webbrowser.open(DATA_DIR.as_uri())

    def closeEvent(self, event) -> None:
        # No popup. Keep .part files.
        for job in self.jobs.values():
            if job.worker is not None and job.worker.isRunning():
                job.worker.cancel()

        deadline = time.time() + 3.0
        for job in self.jobs.values():
            worker = job.worker
            if worker is not None and worker.isRunning():
                remaining = max(0, int((deadline - time.time()) * 1000))
                worker.wait(remaining)

        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
