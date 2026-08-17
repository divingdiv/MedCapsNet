using MedCapsNet, Flux, Random, JLD2

getopt(args, name, default) = (i = findfirst(==(name), args); i === nothing ? default : args[i + 1])
hasflag(args, name) = name in args

build_model(arch::Symbol, n_class::Int; rng) =
    arch === :capsnet  ? CapsNet(n_class; rng) :
    arch === :lenet    ? lenet(n_class) :
    arch === :baseline ? baseline_cnn(n_class) :
    error("unknown arch $arch")

build_lossfn(arch::Symbol) = arch === :capsnet ? capsnet_loss : cnn_loss

function resolve_device(args)
    if getopt(args, "--device", "cpu") == "gpu"
        @eval Main using Metal
        return Flux.gpu
    end
    return identity
end

"""resolve_reclaim(args) -> () -> Nothing

Pairs with `resolve_device`: for `--device gpu` returns a closure to pass as
`train!`'s `reclaim` kwarg, periodically releasing accumulated Metal GPU
memory during long training runs; for `--device cpu` (the default) returns a
no-op, matching `resolve_device`'s `identity` fallback.

Why this is needed: `MtlArray` buffers are freed via Julia GC finalizers, but
a training loop's CPU-side allocation pressure is normally far too low to
trigger Julia's own GC heuristics often enough — unified-memory usage climbs
unboundedly until macOS's jetsam OOM-killer kills the process (observed:
62.78 GB resident on a 24 GB machine, `--device gpu` `capsnet` runs killed
rc=137 after ~15 min). Metal.jl 1.10.2 has no public `reclaim`/pool-trim API
(`isdefined(Metal, :reclaim) == false`); the empirically-verified equivalent
is a full `GC.gc(true)` — which measurably drops
`Metal.device().currentAllocatedSize` by forcing finalizers on abandoned
`MtlArray` wrappers, unlike incremental `GC.gc(false)` or `Metal.synchronize()`
alone, neither of which reliably reclaims device memory at the low relative
memory-pressure levels a single machine hits well before jetsam intervenes —
followed by `Metal.synchronize()` to drain the GPU queue and additionally
invoke Metal's own internal pressure-gated collector as a backstop."""
function resolve_reclaim(args)
    if getopt(args, "--device", "cpu") == "gpu"
        @eval Main using Metal
        return () -> (GC.gc(true); Metal.synchronize())
    end
    return () -> nothing
end

function load_splits(dataset::Symbol, args, rng)
    if dataset === :medical
        dir = getopt(args, "--datadir", "data/diaretdb1")
        tr, val, te = load_medical(dir)
        if val === nothing
            tr, val = split_validation(tr, max(1, MedCapsNet.nobs(tr) ÷ 10); rng)
        end
        return tr, val, te
    end
    full = load_vision(dataset; split=:train)
    tr, val = split_validation(full, 5000; rng)
    return tr, val, load_vision(dataset; split=:test)
end
