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
