using JudiLing
using CSV
using Test

@testset "make cue matrix for latin" begin
  try
    latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
    latin_cue_obj_train = JudiLing.make_cue_matrix(
      latin,
      grams=3,
      words_column=:Word,
      tokenized=false,
      keep_sep=false
      )

    latin_val = latin[101:150,:]
    latin_cue_obj_val = JudiLing.make_cue_matrix(
      latin_val,
      latin_cue_obj_train,
      grams=3,
      words_column=:Word,
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
    words_column=:Syllables,
    tokenized=true,
    sep_token="-",
    keep_sep=true
    )

  french_val = french[101:150,:]
  french_cue_obj_val = JudiLing.make_cue_matrix(
    french_val,
    french_cue_obj_train,
    grams=2,
    words_column=:Syllables,
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
      words_column=:Words,
      tokenized=true,
      sep_token=".",
      keep_sep=true
      )

    utterance_val = utterance[101:150,:]
    utterance_cue_obj_val = JudiLing.make_cue_matrix(
      utterance_val,
      utterance_cue_obj_train,
      grams=3,
      words_column=:Words,
      tokenized=true,
      sep_token=".",
      keep_sep=true
      )

    @test true
  catch e
    @test e == false
  end
end
