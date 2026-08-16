module MedCapsNet

using Flux
using Flux: onehotbatch, DataLoader
using NNlib: softmax, relu
using Optimisers
using Statistics, Random, LinearAlgebra, Printf
using JLD2
using ImageTransformations: imrotate, imresize

include("layers.jl")
include("losses.jl")
# include lines are added task by task:
# include("capsnet.jl")
# include("baselines.jl"); include("data.jl"); include("train.jl")
# include("metrics.jl"); include("visualize.jl")

export squash, safe_norm, margin_loss

end # module
