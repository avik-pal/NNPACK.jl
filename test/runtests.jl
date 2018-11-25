using NNPACK, Test, Random

Random.seed!(144)

@testset "activation" begin
    xs = rand(5,5)

    @test all(sum(softmax(xs), dims = 1) .≈ Float32(1))

    @test sum(softmax(vec(xs))) ≈ Float32(1)
end

@testset "conv2d" begin

    x = reshape(Float32[1:16;], 4, 4, 1, 1)
    w = reshape(Float32[1:9;], 3, 3, 1, 1)

    @test dropdims(conv(x, w, pad=1), dims=(3,4)) ≈ Float32.([
        29   99  207  263
        62  192  372  446
        83  237  417  485
        75  198  330  365])

    x = reshape(Float64[1:20;], 5, 4, 1, 1)
    w = reshape(Float64[1:4;], 2, 2, 1, 1)

    @test dropdims(conv(x, w), dims = (3,4)) ≈ Float32.([
        29  79 129;
        39  89 139;
        49  99 149;
        59 109 159
    ])

    @test_throws Exception dropdims(conv(x, w; stride=2), dims = (3,4)) ≈ Float32.([
        29 49;
        129 149
    ])

    @test dropdims(conv(x, w; pad=1), dims = (3,4)) ≈ Float32.([
        1    9   29   49   48;
        4   29   79  129  115;
        7   39   89  139  122;
        10  49   99  149  129;
        13  59  109  159  136;
        10  40   70  100   80
    ])

    @test size(∇conv_filter(reshape(rand(4,3), 4, 3, 1, 1), x, w)) == size(w)
    @test size(∇conv_data(reshape(rand(4,3), 4, 3, 1, 1), x, w)) == size(x)

end

@testset "maxpool2d" begin

	x = reshape(Float32[1:16;], 4, 4, 1, 1)

	@test dropdims(maxpool(x, (2,2)), dims = (3,4)) ≈ Float32.([
        6.0 14.0;
        8.0 16.0
    ])

    x = reshape(Float64[1:20;], 5, 4, 1, 1)

    @test_throws Exception dropdims(maxpool(x, (2,2)), dims = (3,4)) ≈ Float32.([7 17; 9 19])
    @test_throws Exception dropdims(maxpool(x, (2,2); stride=(2,2)), dims = (3,4)) ≈ Float32.([7 17; 9 19])
    @test_throws Exception dropdims(maxpool(x, (2,2); pad=(1,1)), dims = (3,4)) ≈ Float32.([
        1  11  16;
        3  13  18;
        5  15  20;
    ])

    x = reshape(Float64[1:16;], 4, 4, 1, 1)

    @test dropdims(maxpool(x, (2,2)), dims = (3,4)) == [6 14; 8 16]
    @test dropdims(maxpool(x, (2,2); stride=(2,2)), dims = (3,4)) == [6 14; 8 16]
    @test dropdims(maxpool(x, (3,3); stride=(1,1), pad=(1,1)), dims = (3,4)) == [
         6.0  10.0  14.0  14.0;
         7.0  11.0  15.0  15.0;
         8.0  12.0  16.0  16.0;
         8.0  12.0  16.0  16.0;
    ]

end
