using Test, MedCapsNet

@testset "MedCapsNet" begin
    @test isdefined(MedCapsNet, :MedCapsNet) broken=false
    include("test_layers.jl")
    include("test_losses.jl")
    include("test_capsnet.jl")
    include("test_data.jl")
    include("test_train.jl")
end
