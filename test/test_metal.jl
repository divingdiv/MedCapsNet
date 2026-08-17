using Test, MedCapsNet, Flux, Random, JLD2
using Metal
Metal.allowscalar(false)

@testset "Metal device" begin
    @test Metal.functional()
    rng = Xoshiro(1)
    m = CapsNet(4; rng)
    x = rand(rng, Float32, 28, 28, 1, 6)
    T = Float32.(Flux.onehotbatch(rand(rng, 1:4, 6), 1:4))
    mg, xg, Tg = gpu(m), gpu(x), gpu(T)
    # forward parity
    @test isapprox(cpu(class_lengths(capsules(mg, xg))), class_lengths(capsules(m, x)); atol=1f-3)
    # loss + gradient run without scalar indexing
    l, g = Flux.withgradient(mm -> capsnet_loss(mm, xg, Tg), mg)
    @test isfinite(l) && g[1] !== nothing
    # predict path
    @test predict_classes(m, x; device=gpu) == predict_classes(m, x; device=identity)
    # trained-on-GPU checkpoint is CPU-loadable
    d = LabeledData(rand(rng, Float32, 28, 28, 1, 24), repeat(1:4, 6))
    tr, val = split_validation(d, 8; rng)
    dir = mktempdir()
    train!(capsnet_loss, CapsNet(4; rng), tr, val, 4; epochs=1, batchsize=8, dir, rng, device=gpu)
    st = JLD2.load(joinpath(dir, "best.jld2"))["state"]
    m2 = CapsNet(4; rng)
    Flux.loadmodel!(m2, st)                          # must not throw; state is CPU arrays
    @test all(a -> a isa Array, Flux.trainables(m2))
end
