using Test, MedCapsNet, Flux, Random

@testset "metrics" begin
    C = confusion_matrix([1, 1, 2, 2, 2], [1, 2, 2, 2, 1], 2)
    @test C == [1 1; 1 2]
    rep = classification_report(C)
    @test rep.accuracy ≈ 3 / 5
    @test rep.precision[1] ≈ 1 / 2 && rep.recall[1] ≈ 1 / 2
    @test rep.precision[2] ≈ 2 / 3 && rep.recall[2] ≈ 2 / 3
    @test rep.f1[2] ≈ 2 / 3

    # degenerate column (no predictions for a class) must not divide by zero
    rep0 = classification_report([2 0; 2 0])
    @test rep0.precision[2] == 0.0 && isfinite(rep0.f1[2])

    rng = Xoshiro(5)
    m = CapsNet(3; rng)
    x = rand(rng, Float32, 28, 28, 1, 7)
    preds = predict_classes(m, x; batchsize=4)        # forces 2 batches
    @test length(preds) == 7 && all(p -> p in 1:3, preds)

    chain = Chain(Flux.flatten, Dense(784 => 3))
    @test length(predict_classes(chain, x)) == 7
end
