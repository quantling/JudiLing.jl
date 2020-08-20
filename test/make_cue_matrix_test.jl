using JuLDL
using Test
using SafeTestsets

@testset "make_cue_matrix_test Tests" begin
  @test JuLDL.test_func(1,2) == 4
  @test JuLDL.test_func(4,2) == 10
end