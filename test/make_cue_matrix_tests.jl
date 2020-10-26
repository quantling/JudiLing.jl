using JudiLing
using CSV
using Test

@testset "make cue matrix for latin" begin
  try
    latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
    latin_cue_obj_train = JudiLing.make_cue_matrix(
      latin,
      grams=3,
      target_col=:Word,
      tokenized=false,
      keep_sep=false
      )

    latin_val = latin[101:150,:]
    latin_cue_obj_val = JudiLing.make_cue_matrix(
      latin_val,
      latin_cue_obj_train,
      grams=3,
      target_col=:Word,
      tokenized=false,
      keep_sep=false
      )
    @test true
  catch e
    @test e == false
  end
end

@testset "make cue matrix for french" begin
  try
    french = CSV.DataFrame!(CSV.File(joinpath("data", "french_mini.csv")))
    french_cue_obj_train = JudiLing.make_cue_matrix(
      french,
      grams=2,
      target_col=:Syllables,
      tokenized=true,
      sep_token="-",
      keep_sep=true
      )

    french_val = french[101:150,:]
    french_cue_obj_val = JudiLing.make_cue_matrix(
      french_val,
      french_cue_obj_train,
      grams=2,
      target_col=:Syllables,
      tokenized=true,
      sep_token="-",
      keep_sep=true
      )
      @test true
  catch e
    @test e == false
  end
end

@testset "make cue matrix for utterance" begin
  try
    utterance = CSV.DataFrame!(CSV.File(joinpath("data", "utterance_mini.csv")))
    utterance_cue_obj_train = JudiLing.make_cue_matrix(
      utterance,
      grams=3,
      target_col=:Words,
      tokenized=true,
      sep_token=".",
      keep_sep=true
      )

    utterance_val = utterance[101:150,:]
    utterance_cue_obj_val = JudiLing.make_cue_matrix(
      utterance_val,
      utterance_cue_obj_train,
      grams=3,
      target_col=:Words,
      tokenized=true,
      sep_token=".",
      keep_sep=true
      )

    @test true
  catch e
    @test e == false
  end
end

@testset "make combined cue matrix" begin
  try
    latin_full = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin_mini.csv")))

    latin_train = latin_full[1:3,:]
    latin_val = latin_full[10:15,:]

    cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(
      latin_train,
      latin_val,
      grams=3,
      target_col=:Word,
      tokenized=false,
      keep_sep=false
      )

    @test cue_obj_train.C[1,3] == 1
    @test cue_obj_val.C[1,3] == 0
    @test cue_obj_train.i2f[3] == "oco"
    @test cue_obj_val.i2f[3] == "oco"
    @test cue_obj_train.f2i["oco"] == 3
    @test cue_obj_val.f2i["oco"] == 3
    @test cue_obj_train.A[1,3] == 0
    @test cue_obj_val.A[1,3] == 0
  catch e
    @test false
  end
end