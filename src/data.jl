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

"""Load preprocessed medical data (see scripts/preprocess_*.jl for the writer).
Reference normalization: subtract training mean, then min-max rescale to [0,1]."""
function load_medical(dir::AbstractString)
    f = JLD2.load(joinpath(dir, "data.jld2"))
    μ = f["mean_value"]
    function norm01(a)
        b = a .- μ
        lo, hi = extrema(b)
        return Float32.((b .- lo) ./ (hi - lo + 1f-8))
    end
    train = LabeledData(norm01(f["train_x"]), f["train_y"])
    test  = LabeledData(norm01(f["test_x"]), f["test_y"])
    val   = haskey(f, "val_x") ? LabeledData(norm01(f["val_x"]), f["val_y"]) : nothing
    return train, val, test
end

"""Bounding box of content differing from the [1,1] corner value by > `thresh`
(port of the reference's PIL ImageChops.difference crop). Works on grayscale
matrices; for RGB images pass e.g. `Float32.(green.(img))`."""
function crop_bbox(img::AbstractMatrix; thresh=0.05f0)
    mask = abs.(img .- img[1, 1]) .> thresh
    rows = findall(vec(any(mask; dims=2)))
    cols = findall(vec(any(mask; dims=1)))
    (isempty(rows) || isempty(cols)) && return axes(img, 1), axes(img, 2)
    return first(rows):last(rows), first(cols):last(cols)
end

"""Extract a (2·half)² patch centered at (cy,cx) with ±`jitter` px random
translation, clamped inside bounds, resized to 28×28."""
function extract_patch(g::AbstractMatrix{Float32}, cy::Int, cx::Int;
                       half::Int=100, jitter::Int=30, rng)
    H, W = size(g)
    cy = clamp(cy + rand(rng, -jitter:jitter), half + 1, H - half + 1)
    cx = clamp(cx + rand(rng, -jitter:jitter), half + 1, W - half + 1)
    patch = g[(cy - half):(cy + half - 1), (cx - half):(cx + half - 1)]
    return Float32.(imresize(patch, (28, 28)))
end
