struct LabeledData
    x::Array{Float32,4}                 # (28, 28, C, N)
    y::Vector{Int}                      # 1-based labels
    function LabeledData(x, y)
        size(x, 4) == length(y) || throw(ArgumentError("x/y count mismatch"))
        new(x, y)
    end
end

nobs(d::LabeledData) = length(d.y)
take_obs(d::LabeledData, idx) = LabeledData(d.x[:, :, :, idx], d.y[idx])

function load_vision(name::Symbol; split::Symbol)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    d = name === :mnist   ? MLDatasets.MNIST(; split) :
        name === :fashion ? MLDatasets.FashionMNIST(; split) :
        throw(ArgumentError("unknown dataset $name"))
    x = reshape(Float32.(d.features), 28, 28, 1, :)
    return LabeledData(x, Int.(d.targets) .+ 1)
end

function split_validation(d::LabeledData, n_val::Int; rng)
    p = randperm(rng, nobs(d))
    return take_obs(d, p[(n_val + 1):end]), take_obs(d, p[1:n_val])
end

function _sample_classes(d::LabeledData, keep_fn, rng)
    keep = Int[]
    for c in sort(unique(d.y))
        idx = findall(==(c), d.y)
        n = keep_fn(c, length(idx))
        append!(keep, n == length(idx) ? idx : shuffle(rng, idx)[1:n])
    end
    return take_obs(d, shuffle(rng, keep))
end

"""Stratified subsample keeping `percentage`% of each class (≥ 1 per class)."""
limit_data(d::LabeledData, percentage::Real; rng) =
    _sample_classes(d, (c, n) -> max(1, round(Int, n * percentage / 100)), rng)

"""Downsample only `classes` to `percentage`% of their examples (spec:
unbalance_dict = {percentage, label1, label2}); other classes untouched."""
unbalance_data(d::LabeledData, classes::Vector{Int}, percentage::Real; rng) =
    _sample_classes(d, (c, n) -> c in classes ? max(1, round(Int, n * percentage / 100)) : n, rng)
