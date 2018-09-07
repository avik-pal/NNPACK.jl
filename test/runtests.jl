using NNPACK, Test, Random

@testset "Activation" begin
    @testset "Relu / Leaky Relu" begin
        x = randn(Float32, 4, 4, 4, 4)
        @test relu(x) ≈ max.(x, 0.0f0)
        @test leakyrelu(x, 0.01) ≈ max.(x, 0.01 .* x)

        dy = ones(Float32, 4, 4, 4, 4)
        @test ∇relu(x, dy) ≈ (x .> 0.0f0) .* dy
        @test ∇leakyrelu(x, dy, 0.01) ≈ ((x .> 0.0f0) .+ (x .<= 0.0f0) .* 0.01) .* dy
    end
end

@testset "Maxpool" begin
    x = reshape(Float32[1:36;], 6, 6, 1, 1)

    @test dropdims(maxpool(x, (2,2)), dims = (3,4)) == Float32.([8 20 32; 10 22 34; 12 24 36])
    @test dropdims(maxpool(x, (2,2); stride=(2,2)), dims = (3,4)) == Float32.([8 20 32; 10 22 34; 12 24 36])
    @test dropdims(maxpool(x, (3,3); pad=(1,1), stride=(1,1)), dims = (3,4)) == Float32.([
        8.0  14.0  20.0  26.0  32.0  32.0;
        9.0  15.0  21.0  27.0  33.0  33.0;
        10.0  16.0  22.0  28.0  34.0  34.0;
        11.0  17.0  23.0  29.0  35.0  35.0;
        12.0  18.0  24.0  30.0  36.0  36.0;
        12.0  18.0  24.0  30.0  36.0  36.0;
    ])
end
