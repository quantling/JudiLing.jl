using JudiLing
using Test
using CSV

@testset "make cue matrix for latin" begin
  try
    latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
    JudiLing.make_Yt_matrix(
      2,
      latin,
      words_column=:Word)
    @test true
  catch e
    @test e == false
  end
end