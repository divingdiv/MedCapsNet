using Test, MedCapsNet, Random
using JLD2, Statistics

toy(n; classes=2) = LabeledData(rand(Float32, 28, 28, 1, n),
                                repeat(1:classes, inner=n ÷ classes))

@testset "LabeledData + split" begin
    d = toy(100)
    @test nobs(d) == 100
    @test_throws Exception LabeledData(rand(Float32, 28, 28, 1, 3), [1, 2])
    tr, val = split_validation(d, 20; rng=Xoshiro(1))
    @test nobs(tr) == 80 && nobs(val) == 20
    @test sort(vcat(tr.y, val.y)) == sort(d.y)      # partition, no duplication
end

@testset "limit_data / unbalance_data" begin
    d = toy(200)                                     # 100 per class
    rng = Xoshiro(2)
    lim = limit_data(d, 10; rng)
    @test nobs(lim) == 20
    @test count(==(1), lim.y) == 10 && count(==(2), lim.y) == 10   # stratified

    unb = unbalance_data(d, [1], 20; rng)
    @test count(==(1), unb.y) == 20                  # class 1 → 20%
    @test count(==(2), unb.y) == 100                 # class 2 untouched
end

@testset "augment_data" begin
    d = toy(100)
    rng = Xoshiro(3)
    aug = augment_data(d; fraction=0.05, rng)
    @test nobs(aug) == 105                            # +5% rotated copies
    both = augment_data(d; fraction=0.05, flips=true, rng)
    @test nobs(both) == 110                           # +5% rotated, +5% flipped
    @test all(isfinite, aug.x)
    @test size(aug.x)[1:3] == (28, 28, 1)
end

@testset "load_medical" begin
    dir = mktempdir()
    tx = rand(Float32, 28, 28, 1, 20); vx = rand(Float32, 28, 28, 1, 4)
    ex = rand(Float32, 28, 28, 1, 6)
    jldsave(joinpath(dir, "data.jld2");
            train_x=tx, train_y=repeat([1, 2], 10),
            val_x=vx, val_y=[1, 2, 1, 2],
            test_x=ex, test_y=[1, 2, 1, 2, 1, 2],
            mean_value=Float32(mean(tx)))
    tr, val, te = load_medical(dir)
    @test nobs(tr) == 20 && nobs(val) == 4 && nobs(te) == 6
    @test minimum(tr.x) ≈ 0f0 atol = 1f-6            # min-max normalized
    @test maximum(tr.x) ≈ 1f0 atol = 1f-6
end
