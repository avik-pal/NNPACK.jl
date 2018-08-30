using NNPACK, Test, Random

Random.seed!(144)

#TODO: Add tests and compare the results with NNlib

# @testset "Activation" begin
#     @testset "Relu / Leaky Relu" begin
#         x = randn(Float32, 4, 4, 4, 4)
#         @test relu(x) ≈ max.(x, 0.0f0)
#         @test leakyrelu(x, 0.01) ≈ max.(x, 0.01 .* x)
#
#         dy = ones(Float32, 4, 4, 4, 4)
#         @test ∇relu(x, dy) ≈ (x .> 0.0f0) .* dy
#         @test ∇leakyrelu(x, dy, 0.01) ≈ ((x .> 0.0f0) .+ (x .<= 0.0f0) .* 0.01) .* dy
#     end
# end
