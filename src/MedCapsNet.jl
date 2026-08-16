module MedCapsNet

using Flux
using Flux: onehotbatch, DataLoader
using NNlib: softmax, relu
using Optimisers
using Statistics, Random, LinearAlgebra, Printf
using JLD2
using ImageTransformations: imrotate, imresize
import MLDatasets

include("layers.jl")
include("capsnet.jl")
include("losses.jl")
include("data.jl")
include("train.jl")
# include lines are added task by task:
# include("baselines.jl")
# include("metrics.jl"); include("visualize.jl")

export squash, safe_norm, margin_loss, primary_capsules, prediction_vectors, routing
export CapsNet, capsules, class_lengths, reconstruct, reconstruction_loss, capsnet_loss
export LabeledData, nobs, take_obs, load_vision, split_validation, limit_data, unbalance_data
export augment_data, load_medical
export make_batches, evaluate_loss, train!, cnn_loss

end # module
