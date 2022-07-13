using JudiLing
using CSV
using Test
using SparseArrays
using DataFrames

@testset "cholesky" begin
    C = [1 1 1 1 0 0 0 0; 1 0 1 0 1 0 1 0; 0 0 0 0 1 1 1 1]
    S = [1 0 1 0; 1 1 0 0; 0 0 1 1]

    F = JudiLing.make_transform_matrix(C, S)
    Shat = C * F
    @test -0.05 < sum(Shat-S) < 0.05

    F = JudiLing.make_transform_matrix(C, S, method = :multiplicative)
    Shat = C * F
    @test -0.05 < sum(Shat-S) < 0.05

    C = sparse(C)
    S = Matrix(S)

    F = JudiLing.make_transform_matrix(C, S)
    Shat = C * F
    @test -0.05 < sum(Shat-S) < 0.05

    G = JudiLing.make_transform_matrix(S, C)
    Chat = S * G
    @test -0.1 < sum(Chat-C) < 0.1

    F = JudiLing.make_transform_matrix(C, S, method = :multiplicative)
    Shat = C * F
    @test -0.05 < sum(Shat-S) < 0.05

    G = JudiLing.make_transform_matrix(S, C, method = :multiplicative)
    Chat = S * G
    @test -0.1 < sum(Chat-C) < 0.1

    C = sparse(C)
    S = sparse(S)

    F = JudiLing.make_transform_matrix(C, S)
    Shat = C * F
    @test -0.05 < sum(Shat-S) < 0.05

    F = JudiLing.make_transform_matrix(C, S, method = :multiplicative)
    Shat = C * F
    @test -0.05 < sum(Shat-S) < 0.05

    fac_C = JudiLing.make_transform_fac(C)

    F = JudiLing.make_transform_matrix(fac_C, C, S)
    Shat = C * F
    @test -0.05 < sum(Shat-S) < 0.05

    C = Matrix(C)
    fac_C = JudiLing.make_transform_fac(C)

    F = JudiLing.make_transform_matrix(fac_C, C, S)
    Shat = C * F
    @test -0.05 < sum(Shat-S) < 0.05

    JudiLing.make_transform_matrix(fac_C, C, S,
        output_format = :auto, sparse_ratio = 0.9)

    JudiLing.make_transform_matrix(fac_C, C, S,
        output_format = :sparse)
end
