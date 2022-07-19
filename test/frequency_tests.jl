using JudiLing
using Test
using DataFrames

@testset "frequency" begin
    C = [1 1 1 1 0 0 0 0; 1 0 1 0 1 0 1 0; 0 0 0 0 1 1 1 1]
    S = [1 0 1 0; 1 1 0 0; 0 0 1 1]
    freq = [0, 1, 10]

    F = JudiLing.make_transform_matrix(C, S, freq)

    Shat = C * F
    @test abs(sum(Shat[3,:] - S[3,:])) < abs(sum(Shat[2,:] - S[2,:])) < abs(sum(Shat[1,:] - S[1,:]))
    @test abs(sum(Shat[3,:] - S[3,:])) < 0.01

    freq = [1, 1, 1]
    F = JudiLing.make_transform_matrix(C, S, freq)
    F_orig = JudiLing.make_transform_matrix(C, S)

    @test isapprox(F, F_orig)
end
