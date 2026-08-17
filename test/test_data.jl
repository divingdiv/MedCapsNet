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

@testset "crop_bbox / extract_patch" begin
    img = zeros(Float32, 100, 120)
    img[20:80, 30:90] .= 0.7f0
    rows, cols = crop_bbox(img)
    @test rows == 20:80 && cols == 30:90

    g = rand(Xoshiro(7), Float32, 500, 600)
    p = extract_patch(g, 250, 300; half=100, jitter=30, rng=Xoshiro(7))
    @test size(p) == (28, 28)
    pe = extract_patch(g, 5, 5; half=100, jitter=30, rng=Xoshiro(7))  # near edge: clamped, no error
    @test size(pe) == (28, 28) && all(isfinite, pe)
end

using Colors

@testset "macenko_hematoxylin" begin
    rng = Xoshiro(8)
    # synthetic H&E-ish image: purple-ish and pink-ish regions on white
    img = fill(RGB{Float32}(0.95, 0.95, 0.95), 64, 64)
    img[10:30, 10:30] .= RGB{Float32}(0.4, 0.2, 0.55)   # hematoxylin-like
    img[40:60, 40:60] .= RGB{Float32}(0.85, 0.5, 0.6)   # eosin-like
    h = macenko_hematoxylin(img)
    @test size(h) == (64, 64)
    @test all(0 .≤ h .≤ 1)
    @test mean(h[10:30, 10:30]) > mean(h[40:60, 40:60])  # H region dominates H channel
end
