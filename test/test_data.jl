using Test, MedCapsNet, Random

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
