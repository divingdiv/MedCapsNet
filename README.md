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
