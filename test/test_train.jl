using Test, MedCapsNet, Flux, Random, JLD2

@testset "train!" begin
    rng = Xoshiro(3)
    # separable toy problem: class 2 images are brighter
    x = cat(0.2f0 .* rand(rng, Float32, 28, 28, 1, 40),
            0.5f0 .+ 0.2f0 .* rand(rng, Float32, 28, 28, 1, 40); dims=4)
    d = LabeledData(x, repeat([1, 2], inner=40))
    tr, val = split_validation(d, 16; rng)
    model = Chain(Flux.flatten, Dense(784 => 2; init=Flux.glorot_uniform(rng)))
    dir = mktempdir()
    hist = train!(cnn_loss, model, tr, val, 2; epochs=5, batchsize=16, dir, rng)
    @test length(hist) == 5
    @test hist[end].val_loss < hist[1].val_loss          # it learns
    @test isfile(joinpath(dir, "best.jld2"))
    m2 = Chain(Flux.flatten, Dense(784 => 2))
    Flux.loadmodel!(m2, JLD2.load(joinpath(dir, "best.jld2"))["state"])
    @test evaluate_loss(cnn_loss, m2, val, 2) ≤ minimum(h.val_loss for h in hist) + 1f-5
end

@testset "train! device kwarg" begin
    rng = Xoshiro(3)
    x = cat(0.2f0 .* rand(rng, Float32, 28, 28, 1, 20),
            0.5f0 .+ 0.2f0 .* rand(rng, Float32, 28, 28, 1, 20); dims=4)
    d = LabeledData(x, repeat([1, 2], inner=20))
    tr, val = split_validation(d, 8; rng)
    model = Chain(Flux.flatten, Dense(784 => 2; init=Flux.glorot_uniform(rng)))
    dir = mktempdir()
    hist = train!(cnn_loss, model, tr, val, 2; epochs=2, batchsize=8, dir, rng, device=identity)
    @test length(hist) == 2 && isfile(joinpath(dir, "best.jld2"))
    @test length(predict_classes(model, val.x; device=identity)) == 8
end
