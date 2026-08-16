using Test, MedCapsNet

@testset "squash / safe_norm" begin
    v = reshape(Float32[3, 4], 2, 1)                 # norm 5
    s = squash(v; dims=1)
    @test size(s) == (2, 1)
    @test sqrt(sum(abs2, s)) ≈ 25f0 / 26f0 atol = 1f-4   # ‖s‖²/(1+‖s‖²)
    @test s[1] / s[2] ≈ 0.75f0 atol = 1f-5               # direction preserved
    @test all(isfinite, squash(zeros(Float32, 2, 3); dims=1))
    @test safe_norm(v; dims=1)[1] ≈ 5f0 atol = 1f-3
    big = squash(reshape(Float32[300, 400], 2, 1); dims=1)
    @test sqrt(sum(abs2, big)) < 1f0                     # squashes into unit ball
end
