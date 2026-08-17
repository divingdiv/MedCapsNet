using Test, MedCapsNet

@testset "MedCapsNet" begin
    include("test_layers.jl")
    include("test_losses.jl")
    include("test_capsnet.jl")
    include("test_data.jl")
    include("test_train.jl")
    include("test_metrics.jl")
    include("test_baselines.jl")
    include("test_visualize.jl")
end
