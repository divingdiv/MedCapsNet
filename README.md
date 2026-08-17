# MedCapsNet

A Flux.jl port of a capsule-network (CapsNet) research codebase for medical imaging.

## Status

Package scaffold only — core capsule layers, losses, data loading, training, and
evaluation are being added incrementally as later tasks land.

## Requirements

- Julia >= 1.10
- Flux >= 0.14

## Usage

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using MedCapsNet
```

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Usage

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

`test.jl` takes the same positional `<dataset> <arch>` plus `--seed`,
`--datadir`, and `--modeldir`, and reconstructs the model from the
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

## Experiments

`scripts/run_experiments.jl` automates the paper's three studies — limited
training data, class imbalance, and augmentation — across architectures and
seeds, appending one row per (architecture × configuration × seed) to a CSV:

```bash
julia --project=. scripts/run_experiments.jl mnist --epochs 25 --seeds 1,2,3 \
    --percents 5,10,50,100 --archs capsnet,lenet,baseline --out results.csv
```

Positional argument is the dataset (`mnist`, `fashion`, or `medical`); flags:

- `--epochs N` (default 25)
- `--seeds S1,S2,...` (default `1,2,3`)
- `--percents P1,P2,...` — the limited-data study's percentages (default `5,10,50,100`)
- `--archs A1,A2,...` (default `capsnet,lenet,baseline`)
- `--out PATH` — CSV to append to (default `results.csv`, gitignored; created
  with a header if it doesn't already exist)

For each architecture/seed pair the script runs: one training config per
`--percents` value (study 1: limited data), one config downsampling classes
`1,9` (digits `0`,`8`) to 20% of their examples at 100% data (study 2:
imbalance), and one config with augmentation enabled at 100% data (study 3:
augmentation). Each config trains a fresh model from scratch, reloads the
best checkpoint, and reports test-set accuracy and mean per-class F1.

**Cost warning:** the full 3-arch × 6-config × 3-seed sweep at 25 epochs is
a realistic **hours-on-GPU / days-on-CPU** undertaking — this package's
supported path is CPU-only Flux, so budget accordingly. Start small before
committing to the full sweep, e.g.:

```bash
julia --project=. scripts/run_experiments.jl mnist --seeds 1 --archs capsnet,lenet
```

or the even smaller smoke test used to verify the script itself:

```bash
julia --project=. scripts/run_experiments.jl mnist --epochs 1 --seeds 1 --percents 1 --archs lenet
```

CUDA acceleration is optional and out of scope for this package: if a CUDA
GPU and the CUDA.jl package are available, `using CUDA` followed by
`model = gpu(model)` before training would move the model (and batches) to
the GPU with no other code changes required, but this was not exercised or
tested here — CPU remains the only supported and verified path.
