using JudiLing
using Test
using CSV

@testset "make cue matrix for latin" begin
  try
    latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
    JudiLing.make_Yt_matrix(
      2,
      latin,
      target_col=:Word)
    @test true
  catch e
    @test e == false
  end
end