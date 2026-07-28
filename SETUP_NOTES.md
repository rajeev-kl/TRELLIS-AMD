# TRELLIS-AMD — setup notes for this machine

Verified working end to end on **2026-07-28**: image → Gaussians → mesh → textured GLB.

This machine is not the configuration the repo README targets, and the repo's
`install_amd.sh` does **not** produce a working install here. What follows is what
actually had to change and why.

## Machine

| | Repo README targets | This machine |
| --- | --- | --- |
| GPU | RX 7800 XT, gfx1101 (RDNA3 dGPU) | Radeon 8060S / Ryzen AI MAX+ 395 — **gfx1151** (Strix Halo APU, RDNA3.5) |
| ROCm | 7.2.1 | **7.14** (HIP 7.14.60850), runtime-only install |
| PyTorch | 2.10.0+rocm7.0 | 2.10.0+**rocm7.1** |
| VRAM | 16 GB dedicated GDDR6 | 4 GB UMA carve-out + ~60 GB GTT, shared LPDDR5X |

On the memory question: the 4 GB carve-out is **not** a ceiling. ROCm serves
allocations from GTT, `torch.cuda.get_device_properties(0).total_memory` reports
60.7 GiB, and a 14 GiB allocation succeeds. No BIOS change is needed.

## Scripts

| File | What it does |
| --- | --- |
| `setup_root.sh` | The **only** step needing root. Installs 11 header packages. Idempotent. |
| `prepare_rocm_headers.sh` | Vendors rocPRIM / rocThrust / hipCUB into `.rocm-headers/`. Idempotent. |
| `env_amd.sh` | Sourceable env. **Carries the LD_PRELOAD fix** — see below. |
| `build_extensions.sh` | Builds the three HIP extensions, with header preflight + import verification. |
| `run_app.sh` | Launcher. `./run_app.sh` for the UI, or pass `example.py` / `app_text.py`. |

Normal use: `./run_app.sh` → <http://localhost:7860>

## The critical fix: bundled HSA runtime segfaults on gfx1151

**Every** GPU operation crashed before any TRELLIS code ran — even
`torch.arange(10, device='cuda')` gave SIGSEGV. Backtrace:

```
rocr::AMD::GpuAgent::ReleaseQueueMainScratch(ScratchCache::ScratchInfo&)
rocr::AMD::GpuAgent::QueueCreate(...) [clone .cold]      <-- failure path
rocr::AMD::GpuAgent::CreateInterceptibleQueue(...)
rocr::AMD::GpuAgent::InitDma()
  ... in torch/lib/libhsa-runtime64.so
```

HSA queue creation fails on this APU, and the error-cleanup branch (`.cold`) then
dereferences a null scratch pointer. The bug is in the **ROCm 7.1 HSA runtime
bundled inside the PyTorch wheel**. The system ROCm 7.14 runtime does not have it,
so `env_amd.sh` preloads the system one over the bundled one:

```sh
LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1
```

Ruled out, none of which helped: library-path mixing (`LD_LIBRARY_PATH` was unset
and ROCm is not in `ldconfig`, so torch was already self-contained),
`GPU_MAX_HW_QUEUES=1`, `HSA_ENABLE_SDMA=0`, `HSA_OVERRIDE_GFX_VERSION=11.0.0`,
and disabling the roctracer intercept. Only the runtime swap worked.

`/opt/rocm/lib` is an alternatives symlink into a versioned component dir
(`/opt/rocm/core-7.14` on this machine), so the unversioned path survives ROCm
upgrades. `env_amd.sh` searches `$ROCM_PATH/lib`, `/opt/rocm/lib` and
`/opt/rocm/core-*/lib` in that order, warns loudly if it finds nothing, and can be
overridden with `TRELLIS_HSA_RUNTIME` in `.env`. Locate it manually with:

```sh
find /opt/rocm -name 'libhsa-runtime64.so.1'
```

## Why `install_amd.sh` isn't used

| Its behavior | Why it breaks here |
| --- | --- |
| `pip install torch --index-url .../rocm6.4` | ROCm 6.4 wheels contain no gfx1151 kernels |
| `PYTORCH_ROCM_ARCH=gfx1100` for torchsparse | Wrong arch — would build a binary that can't run |
| assumes thrust / hipCUB present | Not packaged in the Radeon repo at all |

`extensions/diff-gaussian-rasterization/build_hip.sh` needed **no** patch — it
auto-detects the arch from `rocminfo`, which correctly returns `gfx1151`.

## Headers: ROCm here is runtime-only

The installed ROCm ships `.so` files but almost no headers, so all three
extensions failed to compile. Two distinct gaps:

**1. `setup_root.sh` installs 11 packages.** PyTorch's own HIP headers — reached by
every extension via `torch/extension.h` → `ATen/hip/HIPContext.h` — transitively
require hipBLAS, hipBLAS-common, hipSPARSE, hipSOLVER, hipFFT, hipRAND, MIOpen and
RCCL headers, even though the extensions never call those libraries. Plus
`python3.12-dev` (`Python.h`) and `libsparsehash-dev`
(`<google/dense_hash_map>`, for torchsparse's CPU hashmap).

The list was derived by enumerating every ROCm header referenced under
`torch/include`, then **verified by compiling a probe** translation unit
(`torch/extension.h` + `ATen/hip/HIPContext.h` + hipcub + thrust) against the
extracted `.deb`s until clean. The probe caught one the enumeration missed —
`hipblas-common/hipblas-common.h`, which is included *by* `hipblas.h`.

**2. rocPRIM / rocThrust / hipCUB are vendored** in `.rocm-headers/` at tag
`rocm-7.2.4`, because the "ROCm for Radeon" apt repo
(`repo.radeon.com/rocmradeon/apt/26.13`) does not package them at all. They are
header-only. `diff-gaussian-rasterization` needs `hipcub/hipcub.hpp` and
`thrust/scan.h`; torchsparse needs `thrust/device_vector.h`; hipCUB and rocThrust
are both built on rocPRIM.

One wrinkle: each library's `<lib>_version.hpp` is generated by CMake from a
`.hpp.in` template and the headers `#include` it, so cloning alone is not enough.
`prepare_rocm_headers.sh` generates all three from `VERSION_STRING`.

## Code changes made

**`extensions/torchsparse` — `half2 atomicAdd` redefinition.** The fork injects its
own `half2 atomicAdd` shim, but ROCm 7.14 provides `atomicAdd(__half2*, __half2)`
natively in `hip/amd_detail/amd_hip_fp16.h`, which is a hard
`error: redefinition of 'atomicAdd'`. The native version is guarded only by
`__clang__ && __HIP__` with no feature macro to test, so the shim is gated on
`HIP_VERSION < 70300000` instead.

Checked against the ROCm/clr sources to establish the boundary: **absent** in
6.4.2, 7.0.0 and 7.2.1 (so the shim was correct for the ROCm the fork was tested
on), **present** in 7.14. The exact introducing release is somewhere in 7.3–7.13
and was not pinned down — if a ROCm in that range turns out to lack it, widen the
guard. Applied in two places that must stay in sync:

- `torchsparse/backend/convolution/convolution_gather_scatter_hip.hip` (committed,
  compiled directly)
- the `HIP_HALF2_ATOMICADD` template in `setup.py` that regenerates it

**`app_text.py` and `example_text.py` were missing the HIP torchsparse config**
that `app.py` and `example.py` both have — no `GatherScatter` dataflow override, so
torchsparse would use ImplicitGEMM, whose PTX inline assembly cannot run on HIP.
The text-to-3D path in this fork appears to have been untested on AMD. Added the
same block `app.py` uses to both files, plus the AOTriton flag to
`example_text.py`. Text-to-3D is verified working after this — see below.

**`env_amd.sh` exports `PYTHONPATH`** so `import trellis` resolves when running a
driver script kept outside the repo. Python puts the *script's* directory on
`sys.path`, not the cwd, so `./run_app.sh /some/other/path/script.py` otherwise
fails with `ModuleNotFoundError: trellis`.

**Python dependency pins — added to `requirements.txt`.** The file pins
`gradio==4.44.1` (Sept 2024) but left its whole transitive stack unpinned, so a
fresh `pip install` in 2026 resolves libraries that removed APIs this app still
calls. Three separate hard failures, now pinned:

| Pin | Without it |
|---|---|
| `transformers==4.46.3` | 5.x drags in `huggingface_hub` 1.x; TRELLIS targets 4.x |
| `huggingface_hub==0.36.2` | 1.x removed `HfFolder`, which gradio 4.44.1 imports — `import gradio` failed outright |
| `fastapi==0.115.14` | 0.140 pulls starlette 1.x — see below |

The fastapi one is the subtle one. Starlette changed `TemplateResponse` from
`(name, context)` to `(request, name, context)`; older versions sniffed the first
argument and accepted both, but **starlette 1.x removed that shim**. Gradio
4.44.1 still calls the old form in `routes.py`, so starlette bound
`request="frontend/index.html"` and `name={...}`, then handed a dict to Jinja2's
template cache — which needs a hashable key:

```
TypeError: unhashable type: 'dict'
  gradio/routes.py:432  templates.TemplateResponse(template, {...})
  starlette/templating.py  get_template(name)
  jinja2/utils.py  rv = self._mapping[key]
```

Every page render returned **HTTP 500** — the UI was completely unusable while
the backend looked healthy. `fastapi==0.115.14` brings `starlette 0.46.2`, which
still has the shim (verified by inspecting the installed source, not assumed).

`pip install --dry-run -r requirements.txt` now reports nothing to change against
the verified working venv.

**`app.py` launch — was exposing this machine publicly.** Upstream shipped:

```python
demo.launch(server_name="0.0.0.0", share=True)
```

`share=True` opens an outbound tunnel to a public `*.gradio.live` URL (valid 72
hours) pointing at this GPU, and `0.0.0.0` binds every interface so the whole LAN
can reach it. Now defaults to loopback, with explicit opt-in:

```python
server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
share=os.environ.get("GRADIO_SHARE", "0") == "1"
```

Verified: listens on `127.0.0.1:7860`, and a request to the host's LAN IP is
refused. To deliberately share again:
`GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SHARE=1 ./run_app.sh`

Note `app_text.py` used a bare `demo.launch()`, which already defaults to
loopback with no tunnel — it needed no change.

**`gradio_client` patch.** Step 8 of the repo's installer, applied — guards
`get_type` / `_json_schema_to_python_type` against non-dict (boolean) schemas.
Backup at `site-packages/gradio_client/utils.py.orig`.

## Measured results

Full `example.py` run (`assets/example_image/T.png` → GLB):

```
wall clock       3:34   (README quotes 5-10 min for GLB export alone on the 7800 XT)
peak host RSS    9.5 GB
```

| Stage | Result |
| --- | --- |
| Mesh extraction | 328,694 verts / 654,456 faces (README: ~300K/~700K) |
| After postprocessing | 16,380 verts / 32,540 faces (README: ~18K/~36K) |
| GLB | 33,667 verts / 32,540 faces, 1024×1024 texture, 259,010 unique colors |
| Gaussians | 478,336 splats, 17 properties |
| Videos | 300 frames each; 19.6% lit pixels on the Gaussian turntable |

Faster than the README's figure rather than slower, despite shared LPDDR5X.

Each extension was also verified in isolation, because a CPU fallback or a blank
render would still let the pipeline "succeed" while producing garbage:

- **nvdiffrast-hip** — triangle rasterized at 99.2% of analytically expected
  coverage, barycentric interpolation correct. This is the fork's central fix.
- **diff-gaussian-rasterization** — 2048 Gaussians rendered; `SortPairs`,
  `identifyTileRanges` and `renderCUDA` all execute.
- **torchsparse** — **compiled GPU backend** loaded, not the CPU fallback.

## Text-to-3D (`TRELLIS-text-xlarge`, 3.9 GB) — verified working

Runs on gfx1151 after the `app_text.py` / `example_text.py` fix above. Measured on
`"a rusty medieval iron battle axe with a wooden handle"`:

```
load    276.9s   (first run only — fetches the CLIP text encoder)
sample   43.9s
glb      22.6s
```

So **~67 s per asset** steady-state, notably faster than the image path — but only
because the mesh was far simpler, not because the model is faster.

| | Text path (axe) | Image path (test object) |
| --- | --- | --- |
| Faces | 5,597 | 32,540 |
| Slivers (q>10) | 2.9% | 1.9% |
| Texture unique colours | 53,587 | **259,010** |

The ~5× gap in texture colour variety is the clearest quantitative signal of the
appearance-fidelity difference, though it is partly confounded by the simpler
subject.

**Prefer the image path for anything you ship.** The image pipeline conditions on
**DINOv2** patch features (dense spatial information); the text pipeline conditions
on **`CLIPTextModel`** — a single global semantic vector. Text-to-3D has far less
information to reconstruct geometry from. Community adoption agrees: on Hugging
Face, `TRELLIS-image-large` has ~1.97M downloads vs ~8.4K for
`TRELLIS-text-xlarge`, a 235× gap.

### `simplify` is a ratio, not a target — watch this

`to_glb(simplify=0.95)` always removes 95% of faces, so output density tracks input
density. The axe went 141,736 → 5,597 faces; the earlier test object went
654,456 → 32,540. For consistent game assets, derive `simplify` per-asset from a
**target face count** rather than fixing the ratio.

### Input images: what `preprocess_image()` actually does

1. RGBA with real alpha → **your alpha is used directly**; otherwise rembg (u2net).
2. Longest side downscaled to **≤1024**.
3. Auto-crop to subject bbox, **1.2× padding**.
4. Resize to **518×518**.
5. Premultiply by alpha — background becomes black.

Implications: **export RGBA with your own matte** to bypass rembg, which mangles
thin structures (straps, hair, antennae) and thereby corrupts geometry, since shape
follows the silhouette. Don't generate above ~1024px — it is discarded. Prioritise
a clean silhouette over interior micro-detail. Light subjects flat, because
lighting bakes into the albedo permanently (there are no PBR maps).

`run_multi_image()` accepts several views and takes `mode='stochastic'` (default) or
`'multidiffusion'`; multi-view materially beats a single front view, which has to
invent the back.

## Local configuration lives in `.env`, never in tracked files

Machine-specific paths must not be committed — this repo is public. `env_amd.sh`
hardcodes nothing: it auto-detects ROCm and the GPU arch, and reads overrides from
a gitignored `.env`. `.env.example` documents every variable.

| Variable | Purpose |
|---|---|
| `TRELLIS_HF_HOME` | `HF_HOME` — TRELLIS checkpoints + CLIP text encoder |
| `TRELLIS_TORCH_HOME` | `TORCH_HOME` — DINOv2, fetched via `torch.hub` |
| `TRELLIS_HSA_RUNTIME` | Explicit path to the system HSA runtime to preload |
| `ROCM_HOME`, `PYTORCH_ROCM_ARCH` | Override auto-detection |
| `GRADIO_SERVER_NAME/PORT/SHARE` | Server binding; loopback-only by default |

Auto-detection instead of hardcoded values:

- **ROCm dir** — `readlink -f /opt/rocm/include`, then its parent. `/opt/rocm/include`
  is an alternatives symlink into a versioned component dir, so this resolves to the
  right place without naming a version.
- **HSA runtime** — first match of `$ROCM_PATH/lib`, `/opt/rocm/lib`,
  `/opt/rocm/core-*/lib`.
- **GPU arch** — first non-`generic` `gfx*` from `rocminfo`; warns if it isn't gfx1151.
- **ROCm apt package suffix** — `setup_root.sh` derives it from the installed
  `amdrocm-runtime<version>` package rather than hardcoding `7.14`, so it keeps
  working across ROCm upgrades. Override with `ROCM_PKG_SUFFIX=`.
- **Python dev package** — derived from the running interpreter's version.

### Both caches must be set, or DINOv2 silently re-downloads

`HF_HOME` covers `huggingface_hub` only. DINOv2 comes from `torch.hub`
(`trellis_image_to_3d.py`) and needs `TORCH_HOME`. Setting just `HF_HOME` leaves
1.2 GB going back to `~/.cache/torch`.

Total weight footprint is ~10.4 GB across the two caches; none of it is in this repo.

## Any other PyTorch ROCm app on this machine needs the same preload

The HSA segfault is a **PyTorch-wheel** bug, not a TRELLIS one. ComfyUI, A1111 or
raw `diffusers` on this GPU will crash identically on the first GPU op. Each venv
needs:

```sh
export LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1
```

## Known quirks (harmless)

- `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory` on every run —
  the bundled `libdrm_amdgpu` looks under a prefix this system doesn't use. It only
  affects marketing-name lookup.
- The extensions print `[HIP DEBUG]` / `[RasterImpl ctor]` diagnostics. That is the
  fork's own instrumentation, not an error.
- GLB export legitimately loads CPU and GPU heavily for minutes. Not a hang.
