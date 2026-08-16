function make_batches(d::LabeledData, n_class::Int; batchsize::Int,
                      shuffle::Bool=true, rng=Random.default_rng())
    T = Float32.(onehotbatch(d.y, 1:n_class))
    return DataLoader((d.x, T); batchsize, shuffle, rng)
end

function evaluate_loss(lossfn, model, d::LabeledData, n_class; batchsize=256)
    total, count = 0.0, 0
    for (x, T) in make_batches(d, n_class; batchsize, shuffle=false)
        b = size(x, 4)
        total += lossfn(model, x, T) * b
        count += b
    end
    return Float32(total / count)
end

"""Reference trainer: Adam + per-epoch exponential LR decay, best-val checkpoint."""
function train!(lossfn, model, train_d::LabeledData, val_d::LabeledData, n_class::Int;
                epochs=50, batchsize=64, lr=1f-3, decay=0.95f0, weight_decay=0f0,
                dir::AbstractString, rng=Random.default_rng())
    mkpath(dir)
    rule = weight_decay > 0 ? OptimiserChain(WeightDecay(weight_decay), Adam(lr)) : Adam(lr)
    opt = Optimisers.setup(rule, model)
    best = Inf32
    history = NamedTuple[]
    for epoch in 1:epochs
        Optimisers.adjust!(opt; eta=lr * decay^(epoch - 1))
        Flux.trainmode!(model)
        total, count = 0.0, 0
        for (x, T) in make_batches(train_d, n_class; batchsize, rng)
            l, gs = Flux.withgradient(m -> lossfn(m, x, T), model)
            Optimisers.update!(opt, model, gs[1])
            total += l * size(x, 4); count += size(x, 4)
        end
        Flux.testmode!(model)
        vl = evaluate_loss(lossfn, model, val_d, n_class)
        push!(history, (; epoch, train_loss=Float32(total / count), val_loss=vl))
        @info "epoch" epoch train_loss = history[end].train_loss val_loss = vl
        if vl < best
            best = vl
            jldsave(joinpath(dir, "best.jld2"); state=Flux.state(model))
        end
    end
    return history
end
