# MedCapsNet

A Flux.jl port of a capsule-network (CapsNet) research codebase for medical imaging.

## Status

This is a complete Flux.jl port of the reference CapsNet research codebase for
medical imaging: capsule (`CapsNet`) and baseline CNN/LeNet models, dynamic
routing with margin/reconstruction losses, the MNIST/Fashion-MNIST/medical
data pipeline (loading, stratified subsampling, class-imbalance and
augmentation utilities), a shared trainer with checkpointing, confusion-matrix
and precision/recall/F1 metrics, `train.jl`/`test.jl`/`run_experiments.jl`
command-line entry points, reconstruction and latent-dimension-tweak
visualization, and DIARETDB1/TUPAC16 medical-image preprocessing scripts — all
backed by a passing test suite.

## Requirements

- Julia >= 1.10
- Flux — resolved and pinned to 0.16 via this package's `Project.toml`
  `[compat]` entries; `Pkg.instantiate()` picks it up automatically, no manual
  version selection needed.

## Usage

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using MedCapsNet
```

Train and evaluate a model from the command line (run from the repo root so
relative paths like `models/` and `data/` land inside the package):

```bash
julia --project=. scripts/train.jl mnist capsnet --epochs 2 --percentage 1 --seed 1
julia --project=. scripts/test.jl mnist capsnet
julia --project=. scripts/train.jl mnist lenet --epochs 2 --percentage 1 --seed 1
julia --project=. scripts/test.jl mnist lenet
```

`train.jl` takes a positional `<mnist|fashion|medical>` dataset and
`<capsnet|lenet|baseline>` architecture, plus options:

- `--epochs N` (default 50)
- `--percentage P` (default 100)
- `--unbalance L1,L2@P`
- `--augment`
- `--seed S` (default 1)
- `--datadir DIR`
- `--modeldir DIR` (default `models/<dataset>/<arch>/`)
- `--device cpu|gpu` (default `cpu`) — see [GPU (Apple Metal)](#gpu-apple-metal)

`test.jl` takes the same positional `<dataset> <arch>` plus `--seed`,
`--datadir`, `--modeldir`, and `--device`, and reconstructs the model from the
checkpoint saved by `train.jl` to report a confusion matrix and
per-class precision/recall/F1 on the held-out test split.

Three flags support the experiments described in the reference paper:

- `--percentage P` — train on a stratified `P`% subsample of the training
  data (e.g. `--percentage 1` for a 1% smoke run).
- `--unbalance 1,9@20` — downsample only the listed classes to `20`% of
  their examples, leaving the rest untouched. Note labels are **1-based**
  (MNIST/Fashion digit/class `0` is label `1`, digit/class `8` is label `9`),
  so `--unbalance 1,9@20` targets digits `0` and `8`.
- `--augment` — append rotated (and, for Fashion-MNIST, horizontally
  flipped) copies of a fraction of the training samples.

### Visualization

```bash
julia --project=. scripts/visualize.jl mnist [--seed S --modeldir DIR]
```

Loads a trained CapsNet checkpoint (default `models/mnist/capsnet/`, or
`--modeldir DIR`) and the corresponding test split, then writes
`images/reconstructions.png` (five test digits alongside their capsule
reconstructions and predicted labels) and `images/tweak_dim1.png` through
`images/tweak_dim16.png` (one image per class-capsule dimension, showing how
the reconstruction changes as that dimension is swept) — the standard CapsNet
qualitative check that the 16-D pose vector has learned interpretable,
disentangled features.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

This CPU-only suite never loads Metal. The GPU-gated suite is opt-in — see
[GPU (Apple Metal)](#gpu-apple-metal).

## GPU (Apple Metal)

Every script (`train.jl`, `test.jl`, `visualize.jl`, `run_experiments.jl`)
accepts `--device cpu|gpu` (default `cpu`). `--device gpu` loads `Metal.jl`
and moves the model and every batch through `Flux.gpu` (→ `MtlArray`) for
training/evaluation/prediction; checkpoints are always written back to plain
CPU arrays (`Flux.state(cpu(model))`), so a GPU-trained checkpoint loads on
any machine, GPU or not:

```bash
julia --project=. scripts/train.jl mnist lenet --epochs 2 --percentage 1 --device gpu
julia --project=. scripts/test.jl mnist lenet --device gpu
```

Requires a functional Apple Metal GPU (`Metal.functional() == true`); falls
back to ordinary CPU Flux for `--device cpu` (the default) with no `Metal.jl`
load at all.

Metal-specific tests (forward parity, gradient, and checkpoint round-trip
on the GPU) are gated behind an opt-in environment variable, since they
require a Metal-capable Mac and are skipped by default:

```bash
MEDCAPSNET_TEST_METAL=true julia --project=. -e 'using Pkg; Pkg.test()'
```

**Measured speedup:** `julia --project=. scripts/bench_device.jl` times a
warm gradient step (`Flux.withgradient`) and forward pass at the batch sizes
used elsewhere in this repo (capsnet 64, lenet 128), plus `predict_classes`
throughput over 1024 random images, on both devices (7 runs each after 1
warmup; min/median reported given run-to-run spread — see caveat below).
Measured on this machine (Apple Metal, `Metal.functional() == true`):

| device | arch | step ms (min/median) | fwd ms (min/median) | infer ms/1k img (min/median) |
| --- | --- | --- | --- | --- |
| cpu | capsnet | 681.7 / 865.2 | 160.3 / 178.4 | 2796.3 / 2855.8 |
| cpu | lenet | 10.2 / 10.6 | 2.5 / 2.7 | 18.5 / 19.9 |
| gpu | capsnet | 155.1 / 743.0 | 32.2 / 32.6 | 716.6 / 1040.8 |
| gpu | lenet | 4.8 / 9.5 | 1.2 / 1.5 | 4.6 / 5.7 |

CapsNet's gradient step (the number that gates the decision below) is noisy
on GPU at this batch size — the workload is short enough that fixed
dispatch/sync overhead and OS scheduling jitter dominate, so individual runs
swing roughly 2–5x between their min and median. Rerunning the whole script
independently 5 times, GPU's median capsnet step beat CPU's every time
(GPU 452–810 ms vs. CPU 745–934 ms, a 12–50% margin depending on the run) —
never once did GPU lose. Forward-only and LeNet timings are far more stable
on both devices.

**Decision** (per the rule: GPU iff its capsnet gradient step ≤ CPU's, else
STOP): **GPU wins per-step** — but a buffer leak inside Metal.jl's MPS `matmul!`
(the `batched_mul` fast path `prediction_vectors` relies on) makes multi-epoch
**CapsNet GPU training balloon in unified memory until macOS kills it** (observed:
62.8 GB on a 24 GB machine; isolated by per-op bisection — see the trainer's
`reclaim` hook, which fixes the GC-starvation half but cannot reach this leak).
Until that is fixed upstream, **do not train CapsNet with `--device gpu`**: the
project-page runs use CPU for CapsNet (practical after the batched_mul speedup)
and `--device gpu` for LeNet/Baseline, which train bounded (1.75 GB peak, verified).

## Medical datasets

`scripts/preprocess_diaretdb1.jl` builds the `medical` dataset (`data/diaretdb1/data.jld2`)
from the DIARETDB1 standard diabetic retinopathy database — 2-class patches
(class `1` = soft+hard exudates, class `2` = hemorrhages+red small dots)
matching the `load_medical` on-disk contract (`train_x`/`train_y`/`test_x`/
`test_y`/`mean_value`, no validation split — `train.jl`/`test.jl` carve one out
of `train_x` automatically).

1. Download the database package from the official DIARETDB1 project page:
   <https://www.it.lut.fi/project/imageret/diaretdb1/> (a ~141 MB zip
   containing the fundus images, ground truth, and documentation).
2. Extract it so the following layout sits under `raw_data/diaretdb1/`
   (paths the preprocessing script reads directly):

   ```
   raw_data/diaretdb1/
     ddb1_fundusimages/            # the 89 fundus images (*.png)
     ddb1_groundtruth/
       softexudates/  hardexudates/
       hemorrhages/   redsmalldots/
     train_images.txt              # optional — one filename per line
     test_images.txt               # optional — one filename per line
   ```

   If `train_images.txt`/`test_images.txt` are absent, the script falls back
   to the DIARETDB1 evaluation protocol's split of the sorted filenames
   (images 1–28 train, 29–89 test).
3. Run the preprocessing script from the repo root:

   ```bash
   julia --project=. scripts/preprocess_diaretdb1.jl --seed 1
   ```

   This writes `data/diaretdb1/data.jld2` (~480 augmented train patches per
   class, 60/70 test patches), after which `scripts/train.jl` and
   `scripts/test.jl` accept `medical capsnet --datadir data/diaretdb1` like
   any other dataset. Without `raw_data/diaretdb1/ddb1_fundusimages/` present,
   the script exits immediately with a download-instructions error pointing
   back to this section.

`scripts/preprocess_tupac16.jl` builds a second `medical` dataset
(`data/tupac16/data.jld2`) from the TUPAC16 mitosis-detection auxiliary
dataset — 2-class 28×28 patches (class `1` = mitosis, class `2` =
non-mitosis) matching the same `load_medical` on-disk contract, this time
including a validation split (`val_x`/`val_y`).

**Documented deviations:** the reference pipeline stain-normalizes with
Vahadane decomposition (via the `SPAMS` sparse-NMF library). No Julia
package implements sparse NMF/dictionary learning suitable for stain
separation, so this port substitutes **Macenko** stain normalization
(`macenko_hematoxylin`, SVD/eigen-decomposition on the optical-density
plane) — the standard, widely-used alternative to Vahadane for extracting
the hematoxylin channel from H&E-stained histology images. Separately, the
DIARETDB1 preprocessing's `enhance_green` approximates the reference's
OpenCV Lab-space CLAHE (`clipLimit=3.0`) with
`ImageContrastAdjustment.AdaptiveEqualization(clip=0.03)`, which uses a
different clip-limit parameterization and equalizes overall luminance
rather than the Lab `L` channel specifically. And the trainer's batch sizes
(64 for CapsNet, 128 for the CNN baselines) and its 0.95-per-epoch
exponential learning-rate decay rate are this port's own resolution of
trainer hyperparameters the reference leaves underspecified, not values
transcribed from the reference.

1. Download the mitosis-detection auxiliary dataset from the official
   TUPAC16 challenge site: <https://tupac.tue-image.nl/> (registration may
   be required; the auxiliary set is ~656 images from 73 patients with
   centroid-labelled mitoses).
2. Extract it so the following layout sits under `raw_data/tupac16/`
   (paths the preprocessing script reads directly):

   ```
   raw_data/tupac16/
     mitoses_image_data/
       <case>/
         <image>.tif             # 40x HPF tiles
     mitoses_ground_truth/
       <case>/
         <image>.csv             # one "row,col" mitosis center per line
     train.csv                   # optional — one case id per line
     val.csv                     # optional — one case id per line
     test.csv                    # optional — one case id per line
   ```

   If `train.csv`/`val.csv`/`test.csv` are absent, the script falls back to
   a 70/10/20 split of the sorted case directory names.
3. Run the preprocessing script from the repo root:

   ```bash
   julia --project=. scripts/preprocess_tupac16.jl --seed 1
   ```

   Per image, this extracts 30 jittered 100×100→28×28 mitosis-centered
   positive patches (cycling through the image's annotated centers) plus up
   to 3 negative patches from mitosis-free cells of a 3×3 grid, after
   Macenko hematoxylin-channel stain normalization; training-split patches
   are randomly flipped for augmentation. It writes `data/tupac16/data.jld2`
   (train/val/test), after which `scripts/train.jl` and `scripts/test.jl`
   accept `medical capsnet --datadir data/tupac16` like any other dataset.
   Without `raw_data/tupac16/mitoses_image_data/` present, the script exits
   immediately with a download-instructions error pointing back to this
   section.

## Experiments

`scripts/run_experiments.jl` automates the paper's three studies — limited
training data, class imbalance, and augmentation — across architectures and
seeds, appending one row per (architecture × configuration × seed) to a CSV:

```bash
julia --project=. scripts/run_experiments.jl mnist --epochs 25 --seeds 1,2,3 \
    --percents 5,10,50,100 --archs capsnet,lenet,baseline --out results.csv
```

Positional argument is the dataset — **`mnist` or `fashion` only**; the runner
calls `load_vision` unconditionally and hardcodes `n=10` classes and the
`--unbalance 1,9@20` study to MNIST/Fashion-style 10-class label indices, so
`medical` raises an error rather than running. For medical data, drive
`scripts/train.jl medical <arch> --datadir data/diaretdb1` (or
`data/tupac16`) and `scripts/test.jl medical <arch> --datadir ...` directly —
see [Medical datasets](#medical-datasets) above — rather than
`run_experiments.jl`. Flags:

- `--epochs N` (default 25)
- `--seeds S1,S2,...` (default `1,2,3`)
- `--percents P1,P2,...` — the limited-data study's percentages (default `5,10,50,100`)
- `--archs A1,A2,...` (default `capsnet,lenet,baseline`)
- `--out PATH` — CSV to append to (default `results.csv`, gitignored; created
  with a header if it doesn't already exist)
- `--device cpu|gpu` (default `cpu`) — see [GPU (Apple Metal)](#gpu-apple-metal)

For each architecture/seed pair the script runs: one training config per
`--percents` value (study 1: limited data), one config downsampling classes
`1,9` (digits `0`,`8`) to 20% of their examples at 100% data (study 2:
imbalance), and one config with augmentation enabled at 100% data (study 3:
augmentation). Each config trains a fresh model from scratch, reloads the
best checkpoint, and reports test-set accuracy and mean per-class F1.

**Cost warning:** the full 3-arch × 6-config × 3-seed sweep at 25 epochs is
a realistic **hours-on-GPU / days-on-CPU** undertaking, so budget accordingly
— pass `--device gpu` on a Metal-capable Mac (see
[GPU (Apple Metal)](#gpu-apple-metal)) to cut this down substantially. Start
small before committing to the full sweep, e.g.:

```bash
julia --project=. scripts/run_experiments.jl mnist --seeds 1 --archs capsnet,lenet
```

or the even smaller smoke test used to verify the script itself:

```bash
julia --project=. scripts/run_experiments.jl mnist --epochs 1 --seeds 1 --percents 1 --archs lenet
```
