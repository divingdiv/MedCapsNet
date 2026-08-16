using Test, MedCapsNet

@testset "MedCapsNet" begin
    @test isdefined(MedCapsNet, :MedCapsNet) broken=false
    include("test_layers.jl")
    include("test_losses.jl")
end
