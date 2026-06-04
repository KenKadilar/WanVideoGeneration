# WanVideoGeneration: Changelog

Single hand-maintained changelog. Newest first. Versions track the in-app
`APP_VERSION`; "Unreleased" collects work since the last version. Git tracks
file history now, so filenames are no longer version-suffixed.

This workspace was split off from the combined `FluxImageGenerator` workspace
on 2026-05-21 (see "Workspace split" entry below). Detail on the older Flux
features (V1.0 through V1.9 of the combined app) lives in the
[FluxImageGeneration](https://github.com/CanGitArchive/FluxImageGeneration)
repo's `CHANGELOG.md`; those features are summarized as carryover there,
not duplicated here.

## Unreleased

### Workspace split (2026-05-21)

Combined `FluxImageGenerator` workspace split into three:

- **This workspace** kept the existing GitHub repo (renamed to
  `WanVideoGeneration`), the existing git history, the Wan video generator
  (`flux_video_generator.py`), the Wan side of `DATA/` (now at the root, no
  more `FluxImageGenerator` sub-namespace), the `_wan_spike_t2v_1_3b.py` /
  `wan_spike_out/` proof-of-life artifacts, and the
  `CivitaiBatchResumableDownloader.py` tool. Everything Flux-specific
  was deleted from here (it now lives in FluxImageGeneration).
- The new sibling **FluxImageGeneration** workspace got
  `flux_image_generator.py`, `flux_lora_pair_downloader.py`, the Flux side of
  DATA, the `pulid_flux/` parked-research folder, and Flux spike scripts.
  New fresh-init git history; new private GitHub repo at
  `CanGitArchive/FluxImageGeneration`. Its own duplicated `venv\` so the
  two workspaces can update their stacks independently.
- The sibling **LoRaGenerator** workspace absorbed the kohya training stack
  (`train\sd-scripts\`, `venv_train\`, `train\_lora_*.ps1`,
  `train\_prep_face_dataset.py`) since it's the only consumer. Removes the
  cross-workspace coupling that workspace's CLAUDE.md had flagged as a risk.
  Path constants in `LoRA_Maker.py` updated to point at the local copy.
- **`C:\AI_Models\` → NTFS junction → `M:\AI_Models\`** (~333 GB). `C:` had
  filled up under Wan 14B + Flux checkpoints. Linux dual-boot partition on
  Disk 1 was shrunk from 608 GB to 216 GB; the freed ~392 GB became a new
  `M:` drive labeled "Models". All code paths still read/write
  `C:\AI_Models\...`; bytes physically live on `M:`.
- Code-level changes in this workspace:
  - `DATA_DIR = APP_DIR / "DATA"` (dropped the `"FluxImageGenerator"` sub-namespace).
  - `VIDEOS_DIR / VIDEO_REF_FRAMES_DIR / VIDEO_BESTOF_DIR` renamed to
    `Outputs / ReferenceImages / BestOf` (the "Video" prefix made sense when
    they sat next to the Flux app's same-named folders; now they own the
    DATA root, so the prefix is dead weight).
  - `SETTINGS_FILE` / `HISTORY_FILE` renamed `video_settings.json` / `video_gen_history.json` → `settings.json` / `gen_history.json`.
  - `FLUX_OUTPUTS_DIR / FLUX_BESTOF_DIR` now point at the sibling
    `FluxImageGeneration\DATA\Outputs\` / `BestOf\` via a `FLUX_WORKSPACE_DIR`
    absolute path constant near the top of `flux_video_generator.py`.

### Video generator: Wan checkpoint + LoRA support
- **Wan checkpoints** (CivitAI-style `.safetensors` fine-tunes) now load via
  `WanTransformer3DModel.from_single_file(file, config=base/"transformer")` +
  `WanPipeline.from_pretrained(base, transformer=transformer)`, the same two-step
  approach the Flux app uses, with the base model's text encoders / VAE /
  scheduler / config borrowed from the matching architecture's diffusers
  install. On `from_single_file()` failure the log emits a diagnostic (key
  count + top-level prefixes from the safetensors header) so non-standard
  exports can get targeted converters later.
- **Drop-in folder convention:** `C:\AI_Models\Wan_Checkpoints\T2V-1.3B\` and
  `T2V-14B\` (architecturally different, checkpoints can't cross). Folders
  auto-create on launch. Optional `.png`/`.jpg`/`.webp` preview next to each
  file (same convention as Flux LoRAs). Checkpoints appear in the model
  dropdown after the base models tagged `[CKP-1.3B]` / `[CKP-14B]`, inherit
  their base's resolutions/defaults, and are pipeline-kind T2V only for now
  (I2V community fine-tunes are rare).
- **Wan LoRA stack** with reorder / per-LoRA weight / preview thumbnail / add
  / remove / clear-all. Mirrors the Flux LoRA UX. Diffusers-native loader:
  `load_lora_weights` per file → `set_adapters(names, weights)` →
  `fuse_lora()` → `unload_lora_weights()`, applied **before**
  `enable_sequential_cpu_offload()` (offload hooks then only see plain modules,
  the same critical ordering as the Flux app).
- **LoRA folder convention:** `C:\AI_Models\Wan_LoRAs\1.3B\` and
  `C:\AI_Models\Wan_LoRAs\14B\`. The 14B subfolder is shared between
  T2V-14B and I2V-14B since the transformer architecture is the same; the
  LoRA dropdown filters by the selected model's arch and the stack is
  arch-tagged so model-switching drops incompatible entries with a log line.
- **Refresh Models button** in the Model tab rescans both checkpoint and LoRA
  folders without restart. Useful after dropping a CivitAI file in.
- **Sidecar metadata** now records `base_model`, `checkpoint` (full path or
  null), and `lora_stack` (list of `{file, weight}`). Copy-to-Generation in
  the Gallery restores all of it; LoRA files missing from disk are skipped
  with a log line. Settings (`video_settings.json`) persist the stack across
  launches and `isinstance`-validate every field, including the file/weight/arch
  triple.
- **`model_exists()` is checkpoint-aware**: for a `[CKP]` entry, both the
  base model dir AND the checkpoint .safetensors file must be present (and
  the base must not be mid-download per the existing `.incomplete` check).
- **Refactored model registry:** `WAN_BASE_MODELS` (config) + `_MODEL_ENTRIES`
  (runtime, rebuilt by `rebuild_model_registry()`). The dropdown reads from
  `_MODEL_ENTRIES`; every entry carries `kind` (base/checkpoint),
  `pipeline_kind` (t2v/i2v), `arch` (1.3B/14B), `base_path`,
  `checkpoint_path`, resolutions, and defaults, so per-entry logic is one
  lookup instead of two registries.
- **Time-estimator cold-start is now arch-based**, not per-model-name, so
  `[CKP-14B] my_anime.safetensors` correctly uses the 14B baseline (~580 s/step
  scaled by pixel volume) until real history accrues.

### Video generator (new sibling app, `flux_video_generator.py` v0.1)
- **`flux_video_generator.py`**: PyQt6 desktop app for local Wan 2.1 video
  generation on the 6 GB GPU, sibling to `flux_image_generator.py` (shares the
  venv, the `DATA/FluxImageGenerator/` folder, and the `C:\AI_Models\` pool).
  Same dark purple/blue theme, same worker/Gallery patterns. Top tabs:
  Generate / Gallery.
- **One UI, four model options** (model dropdown, missing models tagged
  `[MISSING]` and Generate disabled until the local dir is present): Wan 2.1
  T2V-1.3B (480p, ~40 min/clip on this GPU), T2V-14B (480p/720p), I2V-14B
  (480p variant), I2V-14B (720p variant). Resolution dropdown is
  variant-aware; the I2V "Start Frame" panel is hidden for T2V models.
- **Engine recipe** (from `tools/_wan_spike_t2v_1_3b.py`):
  `WanPipeline` / `WanImageToVideoPipeline` (`from_pretrained`, bf16) +
  `enable_sequential_cpu_offload()` + `vae.enable_tiling()/enable_slicing()`.
  Verified at 2.41 GB peak VRAM for T2V-1.3B. `callback_on_step_end` drives
  the progress bar and gives cooperative cancel.
- **I2V start frame** can be loaded via file picker, drag-drop, Ctrl+V paste,
  or "From Flux Gallery…" which browses the existing
  `DATA/FluxImageGenerator/Outputs/` and `BestOf/` folders (the
  cross-app integration the connector app will hand off through).
  Center-cropped/resized to match the selected output resolution.
- **`model_exists()` is robust to partial downloads**: checks for
  `.incomplete` files under `.cache/huggingface/download/` rather than just
  `model_index.json` (which `snapshot_download` writes early). So a half-pulled
  T2V-14B correctly shows `[MISSING]` and refuses to generate, instead of
  loading a broken pipeline.
- **Outputs land in `DATA/FluxImageGenerator/Videos/`** as `wan_{t2v|i2v}_{seed}_{ts}.mp4`
  plus a `.thumb.jpg` first-frame thumbnail (ffmpeg via `imageio-ffmpeg`) and
  a full `.json` sidecar: model, prompt, negative_prompt, width/height,
  num_frames, fps, duration_seconds, steps, guidance, seed, start_frame,
  elapsed_seconds, timestamp. The sidecar shape matches the connector spec
  in `VIDEO_GEN_CONTINUATION_HANDOFF.md`.
- **Gallery tab** mirrors the image gen's: three sub-tabs (Outputs / Reference
  Frames / Best Of), thumbnail grid + preview + metadata + Copy-to-Generation
  + Use-as-Start-Frame + Delete. Video previews use `QMediaPlayer` +
  `QVideoWidget` (PyQt6.QtMultimedia) with seek slider and play/pause.
  Audio is muted (Wan output is silent).
- **Time estimator** learns per-model s/step from `video_gen_history.json`
  (last 50 runs), normalizes by pixel volume × num_frames, falls back to the
  spike-measured 1.3B baseline scaled by pixel ratio when history is empty.
  "Guess time" button + live countdown timer ("Elapsed: 12m 4s · ~28m 1s left").
- **Settings persisted to `DATA/FluxImageGenerator/video_settings.json`**
  (separate from the image gen's `settings.json`); load uses `isinstance`
  hardening for corrupt values; the start-frame path is restored on launch
  if the file still exists.
- **Misc UI:** Wan-required 4k+1 frame validation with a "try N or N+4"
  hint, taskbar flash on done/fail (`QApplication.alert`), sys.excepthook for
  Qt crashes, "Open Videos folder" button.

### Bug fixes
- **LoRAs now actually apply** *(confirmed working)*. The manual state-dict
  merge only matched LoRAs already named like the diffusers FluxTransformer;
  the dominant CivitAI format is kohya/BFL-native
  (`lora_unet_double_blocks_0_img_attn_qkv`, fused QKV, original BFL
  architecture) and was silently skipped (0 layers merged). New
  `apply_lora_stack()` uses diffusers' native `load_lora_weights()` (proper
  BFL→diffusers remap + QKV/MLP split), then
  `set_adapters`/`fuse_lora`/`unload_lora_weights` so the fused weights stay
  compatible with sequential CPU offload. Manual merge kept as a per-file
  fallback. Adds a `peft` dependency (installed in venv).
- **IP-Adapter applies on Flux-dev without blowing out.** Flux-dev is
  guidance-distilled, so IP-Adapter has ~no effect with only the distilled
  `guidance_scale`; it needs real CFG. When IP-Adapter is active on
  dev/checkpoints we enable `true_cfg_scale` (default **2.0**, tunable via
  `cfg["ip_true_cfg"]`), an empty negative prompt, and a black
  `negative_ip_adapter_image`. ~2x slower (two forward passes/step), only
  when IP is on. **schnell** now skips true CFG entirely; it is
  timestep-distilled and cannot do CFG, so forcing it produced a white
  blowout; XLabs flux-ip-adapter-v2 is a dev model anyway (the log now warns
  on schnell + IP). The earlier `true_cfg=4.0` over-saturated, especially
  with a high IP strength slider; lowered to 2.0 and the log now hints to
  drop IP strength to ~0.5-0.7 if output is blown-out/zoomed.
- **Time estimator is now combination-aware.** V1.9 modeled IP-Adapter as a
  fixed +22 s add-on, but IP-Adapter on dev/checkpoints uses true CFG (two
  forward passes/step) so it ~**doubles** per-step time (multiplicative). It
  also pooled IP and non-IP datapoints together. Result: dev+IP predicted
  ~287 s but actually took ~465 s. Rewrote it (`_combo_sps` /
  `_estimate_seconds_per_image`, shared by the estimate string and the
  countdown): filter history by the exact (model, ip_adapter, img2img)
  combo and learn that combo's real per-step rate; if <2 matching runs
  exist, fall back to the plain rate × `IP_TRUE_CFG_MULT` (1.95, cold-start
  only, skipped for schnell). Removed the old `_extra_overhead` additive
  model. Verified against real history: dev+IP now estimates ~7m33s vs
  ~443-465 s actual; all other combos still accurate.
- **Checkpoint failures are now diagnosable.** On a failed
  `from_single_file()`, the log now reports the file's key count + top-level
  prefixes (header-only probe, fast) and the full error, instead of a guessed
  "ComfyUI format" message. Standard BFL `double_blocks` single-files already
  convert in diffusers 0.38; this surfaces what non-standard variants actually
  are so a targeted converter can be added later. (Loading logic unchanged;
  working checkpoints unaffected.)

### Features
- **Taskbar flash when a generation finishes.** `QApplication.alert()` flashes
  the window/taskbar button on completion and on failure (until the window is
  focused), so you can walk away during long runs.
- **IP True CFG is now a UI knob** (spinbox next to IP Strength, 1.0-8.0,
  persisted). IP Strength and True CFG interact: high True CFG = stronger
  reference but white blowout; low True CFG (1.0-1.5) lets you push IP
  Strength higher. Exposing it lets you find the pair that works for a given
  reference (XLabs flux-ip-adapter-v2 has a narrow usable band) instead of
  fighting a hardcoded value.
- **Gallery tab** (roadmap #1). New top-level tab next to Generate, with
  three sub-tabs (Outputs / Reference Images / Best Of, pointing at
  `DATA\FluxImageGenerator\{Outputs,ReferenceImages,BestOf}`; `BestOf` is
  auto-created on launch). Each sub-tab is a 3-pane splitter: **thumbnail
  grid** (newest first, cap 300, Refresh re-scans) | **large preview**
  (re-scales when you drag the splitter; double-click to fullscreen) |
  **metadata panel** + actions. **Copy to Generation** restores everything
  from the per-image `.json` sidecar: prompt, model, size (added to combo
  if non-default), steps, guidance, seed, sequence length, LoRA stack
  (skips any LoRAs missing from disk), IP-Adapter image + strength + true
  CFG, img2img reference + strength, then jumps to the Generate tab.
  Reference Images tab has no sidecars, so it exposes "Use as img2img
  Reference" and "Use as IP-Adapter Reference" buttons (also auto-enable
  IP-Adapter when used as one). Double-click a thumbnail OR the preview
  pane to open a maximized fullscreen viewer (click anywhere or press Esc
  to close).
- **Per-image metadata enriched** for full Copy-to-Generation: sidecars now
  also store the full IP-Adapter image path, IP true CFG, the img2img
  reference path, and img2img strength (previously dropped, so Copy-to-Gen
  couldn't restore img2img or true CFG). Older sidecars still load; the new
  fields are just absent and skipped.
- **Randomize-seed checkbox** under the Seed field. When checked, the seed
  spinbox becomes read-only (still selectable for Ctrl-C copy) and every
  generation uses a fresh random seed; the seed actually used is written
  back into the field after each image so it's easy to copy and reproduce
  later. Persisted in settings (`randomize_seed`). Hardening: settings
  loader now `isinstance`-checks the loaded seed to ignore corrupt values
  (the empty-string-in-seed crash mode).

### Repo & infrastructure
- Initialized git; project is now version-controlled.
- Renamed `flux_image_generator_V1_9.py` → `flux_image_generator.py` and
  `flux_lora_pair_downloader_V1_4.py` → `flux_lora_pair_downloader.py`.
- Consolidated the 12 per-version `change_log_V1_*.md` files into this single
  `CHANGELOG.md`; removed `version_notes/` and the `combine_version_notes`
  script (no longer needed; one file, edited by hand).
- Deleted the redundant project `models/` folder (was 10 MB of metadata only;
  the real ~54 GB FLUX.1-dev lives in `C:\AI_Models` and is what the code uses).
- Added `.gitignore` (excludes `venv/`, `DATA/`, `models/`, model weights,
  caches, logs).
- `.claude/settings.json` committed with the shared permissive config
  (`defaultMode: bypassPermissions`); `.claude/settings.local.json` stays
  machine-local.
- `tools/` autosave repointed to this project; registered an hourly Windows
  Task Scheduler job that does `git add -A` + commit (no push).

## V1.0 - V1.9 (Flux image generator, pre-split)

Pre-split versions of the combined app were Flux-image-generator-focused.
Their per-version notes live in the
[FluxImageGeneration](https://github.com/CanGitArchive/FluxImageGeneration)
repo's `CHANGELOG.md` under "Carryover features and fixes", not duplicated
here. Wan video generation itself never had a numbered release before the
split (it lived under "Unreleased" in the combined changelog).
