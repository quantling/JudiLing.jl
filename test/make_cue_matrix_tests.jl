using JuLDL
using CSV
using Test

@testset "make cue matrix for latin" begin
  latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
  cue_obj = JuLDL.make_cue_matrix(
    latin,
    grams=3,
    words_column=:Word,
    tokenized=false,
    keep_sep=false
    )
  @test cue_obj.C[11,11] == 0
  @test cue_obj.C[11,8] == 0
  @test cue_obj.C[1,44] == 0
  @test cue_obj.C[56:59,56:59] == [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]
end

# @testset "make cue matrix for french" begin
#   french = CSV.DataFrame!(CSV.File(joinpath("data", "french_mini.csv")))
#   cue_obj = JuLDL.make_cue_matrix(
#     latin,
#     grams=3,
#     words_column=:Word,
#     tokenized=false,
#     keep_sep=false
#     )
#   @test cue_obj.C[11,11] == 0
#   @test cue_obj.C[11,8] == 0
#   @test cue_obj.C[1,44] == 0
#   @test cue_obj.C[56:59,56:59] == [0 0 0; 0 0 0; 0 0 0]
# end
