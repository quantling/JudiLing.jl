using JudiLing
using CSV
using Test
using DataFrames

@testset "eval_SC" begin
  try
    A = [
    1 0 0
    0 1 0
    0 0 1
    1 0 0
    0 1 0
    0 0 1
    1 0 0
    0 1 0
    0 0 1
    0 0 1
    ]

    B = [
    1 0 0
    0 1 0
    0 0 1
    1 0 0
    1 0 0
    0 0 1
    1 0 0
    0 1 0
    0 0 1
    0 0 1
    ]

    C = [
    1 0 0
    0 1 0
    0 0 1
    1 0 0
    1 0 0
    1 0 0
    1 0 0
    1 0 0
    1 0 0
    0 0 1
    ]

    @test JudiLing.eval_SC(A,B) == 0.9
    @test JudiLing.eval_SC(A,C) == 0.6
    @test JudiLing.eval_SC(C,B) == 0.7
    @test JudiLing.eval_SC(A,B,2) == 0.9
    @test JudiLing.eval_SC(A,B,3) == 0.9
    @test JudiLing.eval_SC(A,B,4) == 0.9
    @test JudiLing.eval_SC(A,B,4) == 0.9
    @test JudiLing.eval_SC(A,C,2) == 0.6
    @test JudiLing.eval_SC(C,B,2) == 0.7
  catch
    @test false
  end
end

@testset "eval_SC homophone" begin
  try
    data = DataFrame(A = ["aba", "aba", "abc", "abd"], B = ["M", "F", "F", "M"],
          C = ["S", "L", "S", "L"])

    cue_obj = JudiLing.make_cue_matrix(
      data,
      grams=3,
      target_col=:A,
      tokenized=false,
      keep_sep=false
      )

    n_features = size(cue_obj.C, 2)

    S = JudiLing.make_S_matrix(
      data,
      ["B"],
      ["C"],
      add_noise=true
      )

    G = JudiLing.make_transform_matrix(S, cue_obj.C)

    Chat = S * G
    
    @test JudiLing.eval_SC(Chat, cue_obj.C, data, :A) == 1
    @test JudiLing.eval_SC(Chat, cue_obj.C, data, :A, 2) == 1
    @test JudiLing.eval_SC(Chat, cue_obj.C, data, :A, 3) == 1
  catch
    @test false
  end
end