using Test, MedCapsNet, Flux, Random

@testset "CapsNet" begin
    rng = Xoshiro(1)
    m = CapsNet(10; rng)
    x = rand(rng, Float32, 28, 28, 1, 2)
    v = capsules(m, x)
    @test size(v) == (16, 10, 2)
    lens = class_lengths(v)
    @test size(lens) == (10, 2)
    @test all(0 .< lens .< 1)

    T = Float32.(Flux.onehotbatch([1, 4], 1:10))
    x̂ = reconstruct(m, v, T)
    @test size(x̂) == (784, 2)
    @test all(0 .≤ x̂ .≤ 1)                                # sigmoid output

    @test reconstruction_loss(zeros(Float32, 784, 2), zeros(Float32, 28, 28, 1, 2)) == 0f0
    l = capsnet_loss(m, x, T)
    @test isfinite(l) && l > 0

    g = Flux.gradient(mm -> capsnet_loss(mm, x, T), m)[1]
    @test any(!iszero, g.W)                               # routing path gets gradient
    @test any(!iszero, g.decoder.layers[1].weight)        # decoder path gets gradient
    # (g.decoder[1] is NOT the first Dense's gradient: g.decoder is a NamedTuple
    # with a single field `layers`, so integer-indexing it returns that whole
    # `layers` tuple, not the sub-layer. Use g.decoder.layers[1] instead.)
end
