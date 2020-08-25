using JuLDL
using Test
using CSV

@testset "make cue matrix for latin" begin
  try
    latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
    JuLDL.make_Yt_matrix(
      2,
      latin,
      words_column=:Word)
    @test true
  catch e
    @test e == false
  end
end