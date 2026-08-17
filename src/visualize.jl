_tile!(canvas, img28, row, col) =
    (canvas[(row - 1) * 30 .+ (2:29), (col - 1) * 30 .+ (2:29)] = permutedims(img28))

_save_gray(path, canvas) = FileIO.save(path, Gray.(clamp.(canvas, 0f0, 1f0)))

"""Row 1: inputs; row 2: reconstructions decoded from the *predicted* class
(reference visualize.py figure 1). Returns the predicted labels."""
function reconstruction_grid(m::CapsNet, x, ytrue; path::AbstractString)
    n = size(x, 4)
    ŷ = predict_classes(m, x)
    T = Float32.(onehotbatch(ŷ, 1:m.n_class))
    x̂ = reconstruct(m, capsules(m, x), T)
    canvas = zeros(Float32, 60, 30n)
    for i in 1:n
        _tile!(canvas, x[:, :, 1, i], 1, i)
        _tile!(canvas, reshape(x̂[:, i], 28, 28), 2, i)
    end
    _save_gray(path, canvas)
    return ŷ
end

"""Tweak dimension `dim` of each sample's winning capsule across `steps` values
in ±`radius`, decode each (reference visualize.py figures 2–4)."""
function perturbation_grid(m::CapsNet, x, dim::Int; steps::Int=11,
                           radius::Float32=0.5f0, path::AbstractString)
    n = size(x, 4)
    v = capsules(m, x)
    ŷ = predict_classes(m, x)
    T = Float32.(onehotbatch(ŷ, 1:m.n_class))
    canvas = zeros(Float32, 30n, 30steps)
    for (j, t) in enumerate(range(-radius, radius; length=steps))
        vt = copy(v)
        for i in 1:n
            vt[dim, ŷ[i], i] += t
        end
        x̂ = reconstruct(m, vt, T)
        for i in 1:n
            _tile!(canvas, reshape(x̂[:, i], 28, 28), i, j)
        end
    end
    _save_gray(path, canvas)
end
