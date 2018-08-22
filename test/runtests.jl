using NNPACK, Test, Random

Random.seed!(144)

@testset "Activation" begin
    @testset "Relu / Leaky Relu" begin
        x = randn(Float32, 4, 4, 4, 4)
        @test relu(x) ≈ max.(x, 0.0f0)
        @test leaky_relu(x, 0.01) ≈ max.(x, 0.01 .* x)

        dy = ones(Float32, 4, 4, 4, 4)
        @test ∇relu(x, dy) ≈ (x .> 0.0f0) .* dy
        @test ∇leaky_relu(x, dy, 0.01) ≈ ((x .> 0.0f0) .+ (x .<= 0.0f0) .* 0.01) .* dy
    end
end
