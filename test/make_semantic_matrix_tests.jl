using JudiLing
using CSV
using Test

@testset "make prelinguistic semantic matrix for utterance" begin
  try
    utterance = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "utterance_mini.csv")))
    s_obj_train = JudiLing.make_pS_matrix(utterance)

    utterance_val = utterance[101:end, :]
    s_obj_val = JudiLing.make_pS_matrix(utterance_val, s_obj_train)

    @test true
  catch e
    @test e == false
  end
end

@testset "make semantic matrix for french" begin
  try
    french = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "french_mini.csv")))
    S_train = JudiLing.make_S_matrix(
      french,
      ["Lexeme"],
      ["Tense","Aspect","Person","Number","Gender","Class","Mood"])

    french_val = french[100:end,:]
    S_train, S_val = JudiLing.make_S_matrix(
      french,
      french_val,
      ["Lexeme"],
      ["Tense","Aspect","Person","Number","Gender","Class","Mood"])

    S_train = JudiLing.make_S_matrix(
      french,
      ["Lexeme"])

    S_train, S_val = JudiLing.make_S_matrix(
      french,
      french_val,
      ["Lexeme"])
    @test true
  catch e
    @test e == false
  end
end