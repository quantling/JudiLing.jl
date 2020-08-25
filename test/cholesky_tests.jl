using JuLDL
using CSV
using Test
using SparseArrays

@testset "make transformation matrix for latin" begin
  try
    C = [1 1 1 1 0 0 0 0; 1 0 1 0 1 0 1 0; 0 0 0 0 1 1 1 1]
    S = [1 0 1 0; 1 1 0 0; 0 0 1 1]

    JuLDL.make_transform_matrix(C, S)

    C = sparse(C)
    S = Matrix(S)
    JuLDL.make_transform_matrix(C, S)
    JuLDL.make_transform_matrix(S, C)

    C = sparse(C)
    S = sparse(S)
    JuLDL.make_transform_matrix(C, S)

    fac_C = JuLDL.make_transform_fac(C)
    JuLDL.make_transform_matrix(fac_C, C, S)

    C = Matrix(C)
    fac_C = JuLDL.make_transform_fac(C)
    JuLDL.make_transform_matrix(fac_C, C, S)

    @test true
  catch e
    @test false
  end
end