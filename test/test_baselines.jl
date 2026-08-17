using Test, MedCapsNet, Flux

@testset "baselines" begin
    x = rand(Float32, 28, 28, 1, 3)
    ln = lenet(10)
    @test size(ln(x)) == (10, 3)
    @test sum(length, Flux.trainables(ln)) == 61_706   # classic LeNet-5 count

    bc = baseline_cnn(10)
    @test size(bc(x)) == (10, 3)
    Flux.testmode!(bc)
    @test bc(x) == bc(x)                               # dropout off in test mode
end
