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

"""Append augmented copies of `fraction` of the samples: random rotation in
±`max_angle` degrees (zero-filled, same frame); plus horizontal flips when
`flips=true` (used for Fashion-MNIST). Matches reference data_loader behavior."""
function augment_data(d::LabeledData; fraction=0.05, max_angle=10.0,
                      flips::Bool=false, rng)
    k = max(1, round(Int, fraction * nobs(d)))
    idx = randperm(rng, nobs(d))[1:k]
    rot = similar(d.x, 28, 28, size(d.x, 3), k)
    for (j, i) in enumerate(idx)
        θ = deg2rad((2 * rand(rng) - 1) * max_angle)
        img = @view d.x[:, :, 1, i]
        rotated = Float32.(imrotate(img, θ, axes(img); fillvalue=0f0))
        rot[:, :, 1, j] = replace(x -> isnan(x) ? 0f0 : x, rotated)
    end
    xs, ys = [d.x, rot], [d.y, d.y[idx]]
    if flips
        push!(xs, reverse(d.x[:, :, :, idx]; dims=2))
        push!(ys, d.y[idx])
    end
    x, y = cat(xs...; dims=4), vcat(ys...)
    p = randperm(rng, length(y))
    return LabeledData(x[:, :, :, p], y[p])
end
