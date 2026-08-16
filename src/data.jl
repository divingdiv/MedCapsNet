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
