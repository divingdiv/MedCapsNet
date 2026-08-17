function make_batches(d::LabeledData, n_class::Int; batchsize::Int,
                      shuffle::Bool=true, rng=Random.default_rng())
    T = Float32.(onehotbatch(d.y, 1:n_class))
    return DataLoader((d.x, T); batchsize, shuffle, rng)
end

function evaluate_loss(lossfn, model, d::LabeledData, n_class; batchsize=256, device=identity)
    total, count = 0.0, 0
    for (x, T) in make_batches(d, n_class; batchsize, shuffle=false)
        x, T = device(x), device(T)
        b = size(x, 4)
        total += lossfn(model, x, T) * b
        count += b
    end
    return Float32(total / count)
end

"""Reference trainer: Adam + per-epoch exponential LR decay, best-val checkpoint.

`device` (default `identity`) moves the model and every batch through e.g.
`Flux.gpu`; the checkpoint is always saved as `Flux.state(cpu(model))`, so the
device-local model never leaks — callers reload the trained model from the
checkpoint rather than reusing the `model` argument in place.

`reclaim` (default `() -> nothing`) is called every `reclaim_every` training
batches and once more after each epoch's validation pass. It exists to bound
GPU memory growth during long `device`-backed runs: on Metal, device buffers
are released via Julia GC finalizers, and the CPU-side allocation pressure of
a training loop is normally far too low to trigger Julia's own GC heuristics
often enough, letting unified-memory usage climb until macOS's jetsam
OOM-killer intervenes (observed: 62.78 GB resident on a 24 GB machine). This
function stays Metal-free — the caller (see `scripts/common.jl`'s
`resolve_reclaim`) supplies the actual `GC.gc`/device-sync closure; passing
`reclaim=identity`-style no-ops (the default) keeps CPU runs unaffected.

`reclaim_every` default is `1` (every batch), not a larger stride: measured
on this hardware, per-batch GPU growth for `CapsNet` (which routes through
`NNlib.batched_mul`'s Metal fast path in `prediction_vectors`) is large enough
(~450 MB/batch of GC-collectable churn, plus a further ~113 MB/batch that
survives even an immediate full `GC.gc(true)`) that a stride of 20 lets
gigabytes accumulate between collections; only reclaiming every batch keeps
the collectable portion from compounding. Even at `reclaim_every=1`, a
residual, non-GC-collectable leak remains for `CapsNet` specifically — see
the Task 2 fix-round-1 report for the isolated root cause (`Metal.MPS.matmul!`
inside NNlib's Metal `batched_mul` path) and why it is not fixed here."""
function train!(lossfn, model, train_d::LabeledData, val_d::LabeledData, n_class::Int;
                epochs=50, batchsize=64, lr=1f-3, decay=0.95f0, weight_decay=0f0,
                dir::AbstractString, rng=Random.default_rng(), device=identity,
                reclaim=() -> nothing, reclaim_every::Int=1)
    mkpath(dir)
    model = device(model)
    rule = weight_decay > 0 ? OptimiserChain(WeightDecay(weight_decay), Adam(lr)) : Adam(lr)
    opt = Optimisers.setup(rule, model)
    best = Inf32
    history = NamedTuple[]
    batch_count = 0
    for epoch in 1:epochs
        Optimisers.adjust!(opt; eta=lr * decay^(epoch - 1))
        Flux.trainmode!(model)
        total, count = 0.0, 0
        for (x, T) in make_batches(train_d, n_class; batchsize, rng)
            x, T = device(x), device(T)
            l, gs = Flux.withgradient(m -> lossfn(m, x, T), model)
            Optimisers.update!(opt, model, gs[1])
            total += l * size(x, 4); count += size(x, 4)
            batch_count += 1
            batch_count % reclaim_every == 0 && reclaim()
        end
        Flux.testmode!(model)
        vl = evaluate_loss(lossfn, model, val_d, n_class; device)
        reclaim()                                          # belt and braces
        push!(history, (; epoch, train_loss=Float32(total / count), val_loss=vl))
        @info "epoch" epoch train_loss = history[end].train_loss val_loss = vl
        if vl < best
            best = vl
            jldsave(joinpath(dir, "best.jld2"); state=Flux.state(cpu(model)))
        end
    end
    return history
end
