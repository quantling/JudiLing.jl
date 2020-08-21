using JuLDL
using CSV
using Test

@testset "make cue matrix for latin" begin
  latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
  latin_cue_obj = JuLDL.make_cue_matrix(
    latin,
    grams=3,
    words_column=:Word,
    tokenized=false,
    keep_sep=false
    )
  @test latin_cue_obj.C[11,11] == 0
  @test latin_cue_obj.C[11,8] == 0
  @test latin_cue_obj.C[1,44] == 0
  @test latin_cue_obj.f2i["sis"] == 158
  @test latin_cue_obj.i2f[134] == "sti"
  @test latin_cue_obj.C[1:4,5:9] == [1 0 0 0 0; 0 1 1 1 1; 0 1 0 0 0; 0 1 1 0 0]
end

@testset "make cue matrix for french" begin
  french = CSV.DataFrame!(CSV.File(joinpath("data", "french_mini.csv")))
  french_cue_obj = JuLDL.make_cue_matrix(
    french,
    grams=2,
    words_column=:Syllables,
    tokenized=true,
    sep_token="-",
    keep_sep=true
    )
  @test french_cue_obj.C[11,11] == 0
  @test french_cue_obj.C[11,8] == 1
  @test french_cue_obj.C[1,44] == 0
  @test french_cue_obj.f2i["t§-#"] == 131
  @test french_cue_obj.i2f[207] == "N°-R§"
  @test french_cue_obj.C[1:4,5:9] == [0 0 0 0 0; 1 1 1 0 0; 0 0 0 1 1; 0 0 0 0 0]
end

@testset "make cue matrix for utterance" begin
  utterance = CSV.DataFrame!(CSV.File(joinpath("data", "utterance_mini.csv")))
  utterance_cue_obj = JuLDL.make_cue_matrix(
    utterance,
    grams=3,
    words_column=:Words,
    tokenized=true,
    sep_token=".",
    keep_sep=true
    )
  @test utterance_cue_obj.C[11,11] == 0
  @test utterance_cue_obj.C[11,8] == 0
  @test utterance_cue_obj.C[1,44] == 0
  @test utterance_cue_obj.f2i["#.服务员.我们"] == 176
  @test utterance_cue_obj.i2f[285] == "今天.不.是"
  @test utterance_cue_obj.C[1:4,5:9] == [1 1 0 0 0; 0 0 1 1 1; 0 1 0 0 0; 0 0 0 0 0]
end
