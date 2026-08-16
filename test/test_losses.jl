using Test, MedCapsNet

@testset "margin_loss" begin
    T = Float32[1 0; 0 1]                       # 2 classes, 2 samples, one-hot
    perfect = Float32[1 0; 0 1]
    @test margin_loss(perfect, T) ≈ 0f0 atol = 1f-6
    worst = Float32[0 1; 1 0]
    # per sample: 1·(0.9)² + 0.5·(1−0.1)² = 0.81 + 0.405 = 1.215
    @test margin_loss(worst, T) ≈ 1.215f0 atol = 1f-5
    # inside the margins (length 0.9 for true, 0.1 for others) → zero loss
    inside = Float32[0.95 0.05; 0.05 0.95]
    @test margin_loss(inside, T) ≈ 0f0 atol = 1f-6
end
