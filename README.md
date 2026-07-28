# TRELLIS for AMD Strix Halo — Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151)

**Image-to-3D and text-to-3D asset generation on an AMD Strix Halo APU with ROCm.**

This fork targets **one specific chipset: `gfx1151`** — the RDNA 3.5 integrated GPU in
AMD's Strix Halo APUs (Ryzen AI MAX / MAX+ series, Radeon 8050S/8060S Graphics). It is
a fork of [CalebisGross/TRELLIS-AMD](https://github.com/CalebisGross/TRELLIS-AMD),
which ported [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) to AMD discrete
cards (RX 7800 XT, `gfx1101`).

> **Status: verified working end-to-end on gfx1151 (2026-07-28).**
> Image → 3D Gaussians → mesh → textured GLB, plus text-to-3D. Full run in
> **3m15s**, peak host RSS 8.8 GB. Every fix in this fork was validated by an
> actual generation, not just an import check.

If you have a **discrete** Radeon (RX 7000/9000 series), use the
[parent fork](https://github.com/CalebisGross/TRELLIS-AMD) instead — this one carries
APU-specific workarounds you don't need.

---

## Is this your hardware?

Run either command:

```bash
rocminfo | grep -m1 gfx                  # -> gfx1151
rocm-smi --showproductname               # -> Radeon 8060S Graphics / SKU STRXLGEN
```

If you see `gfx1151`, yes.

### Chipset identification

This silicon is sold and referred to under many names, all of which are the same GPU:

| Field | Value |
|---|---|
| **LLVM / ROCm target** | `gfx1151` |
| **Architecture** | RDNA 3.5 (GFX11.5) |
| **Codename** | Strix Halo (`STRXLGEN`) |
| **GPU marketing name** | AMD Radeon 8060S Graphics (also 8050S / 8040S on lower SKUs) |
| **APU marketing names** | Ryzen AI MAX+ 395, Ryzen AI MAX 390 / 385 (Ryzen AI MAX 300 series) |
| **PCI device ID** | `1002:1586` (rev `c1`) |
| **KFD `gfx_target_version`** | `110501` |
| **Compute units** | 40 CU = 20 WGP = 80 × SIMD32 |
| **Memory model** | Unified/shared system memory (UMA). Small VRAM carve-out + large GTT |
| **Kernel driver** | `amdgpu` (verified on amdgpu 3.64.0, Linux 7.0.0) |

Values above were read from `rocminfo`, `rocm-smi`, and
`/sys/devices/virtual/kfd/kfd/topology/nodes/*/properties` on the test machine.

**Related but untested targets:** `gfx1150` (Strix Point — Radeon 890M, Ryzen AI 9 HX 370)
and `gfx1152` (Krackan Point) are the same RDNA 3.5 family. The fixes here are likely
relevant, but nothing in this fork has been tested on them.

**Search terms:** TRELLIS AMD APU · TRELLIS Strix Halo · gfx1151 ROCm PyTorch ·
Ryzen AI MAX 395 machine learning · Radeon 8060S image to 3D · RDNA 3.5 ROCm ·
Strix Halo 3D generation · TRELLIS ROCm 7 · AMD APU unified memory PyTorch

---

## Verified configuration

| Component | Version |
|---|---|
| GPU | Radeon 8060S / Ryzen AI MAX+ 395, `gfx1151` |
| ROCm | **7.14** (HIP 7.14.60850) — runtime-only install, from `repo.radeon.com/rocmradeon/apt/26.13` |
| PyTorch | **2.10.0+rocm7.1** (`gfx1151` confirmed in `torch.cuda.get_arch_list()`) |
| Python | 3.12 |
| OS | Linux Mint 22.3 / Ubuntu 24.04 noble, kernel 7.0.0 |
| System RAM | 121 GB (GPU sees 60.7 GiB) |

### On memory: the VRAM carve-out is not a ceiling

`rocm-smi` reports only **4 GB** of "VRAM" on this APU (the BIOS UMA carve-out), which
looks fatally short of TRELLIS's documented 16 GB requirement. It isn't a limit:

```
torch.cuda.get_device_properties(0).total_memory  ->  60.7 GiB
```

ROCm serves allocations from GTT, and a **14 GiB single allocation succeeds**. No BIOS
change is needed. This is the main reason a Strix Halo APU is a genuinely good fit for
3D generation — you get workstation-class capacity on an integrated GPU.

---

## Quick start

```bash
git clone https://github.com/rajeev-kl/TRELLIS-AMD
cd TRELLIS-AMD

cp .env.example .env          # optional: cache locations etc. (gitignored)
sudo ./setup_root.sh          # ONLY privileged step — installs 11 header packages
./prepare_rocm_headers.sh     # vendors rocPRIM / rocThrust / hipCUB (header-only)
./setup_venv.sh               # venv + PyTorch ROCm + deps + gradio_client patch
./build_extensions.sh         # nvdiffrast-hip, diff-gaussian-rasterization, torchsparse
./run_app.sh                  # http://127.0.0.1:7860
```

Other entry points:

```bash
./run_app.sh example.py       # headless image -> GLB
./run_app.sh app_text.py      # text-to-3D UI
./run_app.sh example_text.py  # headless text -> GLB
```

Every script is idempotent and re-runnable. `setup_root.sh` is the only one needing
root, and it installs **headers only** — it does not touch ROCm, the driver, or kernel
parameters.

> **Do not use upstream's `install_amd.sh` on this hardware.** It installs ROCm 6.4
> wheels (no `gfx1151` kernels) and hardcodes `PYTORCH_ROCM_ARCH=gfx1100` for
> torchsparse, producing a build that cannot run here.

---

## Measured results

Full `example.py` run, `assets/example_image/T.png` → textured GLB:

```
wall clock       3m15s
peak host RSS    8.8 GB
```

| Stage | Output |
|---|---|
| Mesh extraction | 328,707 verts / 654,460 faces |
| After postprocessing | 16,412 verts / 32,577 faces |
| Textured GLB | 33,563 verts / 32,577 faces, 1024² texture, 256,208 unique colours |
| Gaussians | 478,336 splats (.ply) |

Text-to-3D (`TRELLIS-text-xlarge`), steady state: **~67 s/asset** (44 s sampling +
23 s GLB) on a simple prop.

Each extension was also verified in isolation, because a CPU fallback or a blank render
would still let the pipeline "succeed" while producing garbage:

- **nvdiffrast-hip** — triangle rasterised at 99.2% of analytically expected coverage
- **diff-gaussian-rasterization** — 2048 Gaussians rendered, HIP kernels executing
- **torchsparse** — **compiled GPU backend** loaded, not the CPU fallback

---

## What this fork changes, and why

Full reasoning for every item is in **[SETUP_NOTES.md](SETUP_NOTES.md)**.

### 1. The bundled HSA runtime segfaults on gfx1151 — the critical fix

Before any TRELLIS code runs, **every** GPU operation crashes. Even
`torch.arange(10, device='cuda')` gives SIGSEGV:

```
rocr::AMD::GpuAgent::ReleaseQueueMainScratch(ScratchCache::ScratchInfo&)
rocr::AMD::GpuAgent::QueueCreate(...) [clone .cold]      <-- failure path
rocr::AMD::GpuAgent::CreateInterceptibleQueue(...)
rocr::AMD::GpuAgent::InitDma()
  ... in torch/lib/libhsa-runtime64.so
```

HSA queue creation fails on this APU and the error-cleanup branch dereferences a null
scratch pointer. The bug is in the **ROCm HSA runtime bundled inside the PyTorch
wheel**; the system ROCm 7.14 runtime does not have it. `env_amd.sh` preloads the
system one:

```sh
export LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1
```

**This is not TRELLIS-specific.** Any PyTorch ROCm application on this chipset —
ComfyUI, A1111, plain `diffusers` — will crash the same way and needs the same preload.

Ruled out and ineffective: `GPU_MAX_HW_QUEUES=1`, `HSA_ENABLE_SDMA=0`,
`HSA_OVERRIDE_GFX_VERSION=11.0.0`, disabling the roctracer intercept, and library-path
mixing. Only the runtime swap works.

### 2. `half2 atomicAdd` redefinition in torchsparse

ROCm 7.14 provides `atomicAdd(__half2*, __half2)` natively, colliding with the shim
torchsparse injects — a hard `error: redefinition of 'atomicAdd'`. Now gated on
`HIP_VERSION < 70300000`. Checked against ROCm/clr sources: absent in 6.4.2, 7.0.0 and
7.2.1; present in 7.14.

### 3. ROCm-for-Radeon ships almost no headers

The apt repo used for this hardware is runtime-focused, so all three extensions fail to
compile. `setup_root.sh` installs 11 packages: PyTorch's own HIP headers transitively
require hipBLAS, hipBLAS-common, hipSPARSE, hipSOLVER, hipFFT, hipRAND, MIOpen and RCCL
headers even though the extensions never call those libraries — plus `python3.12-dev`
and `libsparsehash-dev`.

rocPRIM / rocThrust / hipCUB **aren't packaged at all** in that repo, so
`prepare_rocm_headers.sh` vendors them (header-only, tag `rocm-7.2.4`) and generates the
CMake-templated `<lib>_version.hpp` files the headers `#include`.

### 4. Text-to-3D was untested on AMD upstream

`app_text.py` and `example_text.py` lacked the HIP torchsparse configuration that
`app.py` and `example.py` have, so torchsparse would use ImplicitGEMM — whose PTX inline
assembly cannot run on HIP. Added; text-to-3D now works.

### 5. Dependency pins

`requirements.txt` pinned `gradio==4.44.1` but left its entire transitive stack
unpinned, so a fresh install resolves 2026 libraries against a 2024 app. Three hard
failures, now pinned: `transformers==4.46.3`, `huggingface_hub==0.36.2` (1.x removed
`HfFolder`, which gradio imports), `fastapi==0.115.14` (0.140 pulls starlette 1.x, which
removed the legacy `TemplateResponse` shim → every page render 500s).

### 6. The Gradio app no longer exposes your machine publicly

Upstream shipped `demo.launch(server_name="0.0.0.0", share=True)` — binding all
interfaces **and** opening a public `*.gradio.live` tunnel to your GPU. Now loopback by
default:

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SHARE=1 ./run_app.sh   # deliberate opt-in
```

---

## Troubleshooting

Error strings, so searches land here:

| Symptom | Cause / fix |
|---|---|
| SIGSEGV on any GPU op; `ReleaseQueueMainScratch` in the backtrace | Missing `LD_PRELOAD` of the system HSA runtime — see §1. Use `./run_app.sh` or `source env_amd.sh` |
| `fatal error: hip/hip_runtime.h: No such file or directory` | `sudo ./setup_root.sh` |
| `fatal error: hipsparse/hipsparse.h: No such file or directory` | Same — pulled in by PyTorch's own headers |
| `fatal error: Python.h: No such file or directory` | Same — needs `python3.12-dev` |
| `error: redefinition of 'atomicAdd'` | ROCm ≥ 7.3 with an unguarded shim — see §2 |
| `TypeError: unhashable type: 'dict'` from Jinja2 / HTTP 500 on the UI | starlette too new — `fastapi==0.115.14` |
| `ImportError: cannot import name 'HfFolder'` | `huggingface_hub` 1.x — pin to `0.36.2` |
| `ModuleNotFoundError: No module named 'trellis'` | Run via `./run_app.sh`, which puts the repo on `PYTHONPATH` |
| App opens a random `*.gradio.live` URL | Old `share=True` default — see §6 |
| `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file` | Harmless; affects marketing-name lookup only |
| GLB export appears to hang for minutes | Expected — texture baking is 2500 optimisation steps |

`env_amd.sh` auto-detects this path and warns if it cannot find it. Override via
`TRELLIS_HSA_RUNTIME` in `.env` (see `.env.example`). Locate it manually with:

```bash
find /opt/rocm -name 'libhsa-runtime64.so.1'
```

---

## Known limitations

Inherited from the parent fork:

1. **Coarse rasteriser is serialised** and slower than NVIDIA's warp-parallel version.
2. **~7% silent triangle culls** from the `triHeader[i].misc` bounds-check fix — see
   [experiments/raster/findings.md](experiments/raster/findings.md).
3. **`fill_holes` uses 100 views, not upstream's 1000**, with poles clamped so
   degenerate view matrices don't hang the HIP rasteriser.

Specific to output quality for game use:

4. **No PBR maps.** Output is base colour only, with the input image's lighting baked
   in — it won't relight correctly in-engine.
5. **Topology is triangle soup.** Auto-generated, no edge loops; ~1.9% slivers measured.
   Fine for static props, not riggable for animated characters without retopology.
6. **`simplify` is a ratio, not a target.** `simplify=0.95` always removes 95% of faces,
   so output density tracks input density.

---

## Scripts

| File | Purpose |
|---|---|
| `setup_root.sh` | The only privileged step. Installs 11 header packages, verifies each landed |
| `prepare_rocm_headers.sh` | Vendors rocPRIM / rocThrust / hipCUB into `.rocm-headers/` |
| `setup_venv.sh` | venv, PyTorch ROCm (asserts `gfx1151`), deps, `gradio_client` patch |
| `build_extensions.sh` | Builds the three HIP extensions with header preflight + import verification |
| `env_amd.sh` | Sourceable environment. **Carries the `LD_PRELOAD` fix** |
| `run_app.sh` | Launcher — `./run_app.sh [script.py]` |
| `SETUP_NOTES.md` | Full reasoning, backtraces, and measurements |

---

## Credits

- [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) — original model and code (MIT)
- [CalebisGross/TRELLIS-AMD](https://github.com/CalebisGross/TRELLIS-AMD) — the AMD/HIP
  port this builds on, including the multi-month rasteriser investigation that made any
  of this possible
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast) — NVIDIA
- rocPRIM / rocThrust / hipCUB — AMD ROCm (vendored, header-only)

Model weights (`microsoft/TRELLIS-image-large`, `microsoft/TRELLIS-text-xlarge`) are
MIT-licensed, which permits commercial use — relevant if you're generating game assets.

## License

See the original licenses for TRELLIS, nvdiffrast, and
diff-gaussian-rasterization. This fork adds no new licence terms.
