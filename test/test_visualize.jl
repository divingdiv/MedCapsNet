using Test, MedCapsNet, Random

@testset "visualize" begin
    rng = Xoshiro(6)
    m = CapsNet(4; rng)
    x = rand(rng, Float32, 28, 28, 1, 5)
    dir = mktempdir()
    p1 = joinpath(dir, "recon.png")
    preds = reconstruction_grid(m, x, [1, 2, 3, 4, 1]; path=p1)
    @test isfile(p1) && length(preds) == 5
    p2 = joinpath(dir, "tweak_dim3.png")
    perturbation_grid(m, x, 3; path=p2)
    @test isfile(p2)
end
