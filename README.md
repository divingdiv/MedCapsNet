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
