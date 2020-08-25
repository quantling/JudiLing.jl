using JuLDL
using Test

@testset "make cue matrix for latin" begin
  try
    i2f = Dict([(1, "#ab"), (2, "abc"), (3, "bc#"), (4, "#bc"), (5, "ab#")])
    JuLDL.make_adjacency_matrix(i2f)
    @test true
  catch e
    @test false
  end
end