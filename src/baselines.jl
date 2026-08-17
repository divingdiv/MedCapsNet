"""LeNet (reference lenet.py): input zero-padded to 32×32, truncated-normal σ=0.1."""
function lenet(n_class::Int; channels::Int=1)
    init(dims::Integer...) = Flux.truncated_normal(dims...; std=0.1f0)
    return Chain(
        x -> NNlib.pad_zeros(x, (2, 2, 2, 2); dims=(1, 2)),
        Conv((5, 5), channels => 6, relu; init),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu; init),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(400 => 120, relu; init),
        Dense(120 => 84, relu; init),
        Dense(84 => n_class; init),
    )
end

"""Baseline CNN from Sabour et al. (reference baseline.py). FC layers are 16×16
and 1×1 convolutions. Train with weight_decay = 5f-4 (L2, reference value)."""
function baseline_cnn(n_class::Int; channels::Int=1)
    return Chain(
        Conv((5, 5), channels => 256, relu),   # 28 → 24
        Conv((5, 5), 256 => 256, relu),        # 24 → 20
        Conv((5, 5), 256 => 128, relu),        # 20 → 16
        Conv((16, 16), 128 => 328, relu),      # 16 → 1
        Conv((1, 1), 328 => 192, relu),
        Dropout(0.5),
        Conv((1, 1), 192 => n_class),
        Flux.flatten,
    )
end
