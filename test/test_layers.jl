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

using Flux

@testset "primary_capsules" begin
    conv1 = Conv((9, 9), 1 => 256, relu)
    conv2 = Conv((9, 9), 256 => 256, relu; stride=2)
    x = rand(Float32, 28, 28, 1, 2)
    u = MedCapsNet.primary_capsules(conv1, conv2, x)
    @test size(u) == (8, 1152, 2)
    @test all(safe_norm(u; dims=1) .< 1f0)      # squashed
end
