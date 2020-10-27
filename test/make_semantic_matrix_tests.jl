using JudiLing
using CSV
using Test
using DataFrames

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

@testset "make semantic matrix for lexome" begin
  try
    latin = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin_mini.csv")))
    latin_val = latin[1:20,:]

    L1 = JudiLing.make_L_matrix(
      latin,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      ncol=20)

    L2 = JudiLing.make_L_matrix(
      latin,
      ["Lexeme"],
      ncol=20)

    S1 = JudiLing.make_S_matrix(
      latin,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      L1,
      add_noise=true,
      sd_noise=1,
      normalized=false
      )

    S1 = JudiLing.make_S_matrix(
      latin,
      ["Lexeme"],
      L1,
      add_noise=true,
      sd_noise=1,
      normalized=false
      )

    S1, S2 = JudiLing.make_S_matrix(
      latin,
      latin_val,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      L1,
      add_noise=true,
      sd_noise=1,
      normalized=false
      )

    S1, S2 = JudiLing.make_S_matrix(
      latin,
      latin_val,
      ["Lexeme"],
      L1,
      add_noise=true,
      sd_noise=1,
      normalized=false
      )

    @test true
  catch e
    @test e == false
  end
end

@testset "make combined semantic matrix" begin
  try
    latin_full = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin_mini.csv")))

    latin_train = latin_full[1:3,:]
    latin_val = latin_full[10:15,:]

    S_train, S_val = JudiLing.make_combined_S_matrix(
      latin_train,
      latin_val,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      ncol=5)

    # @test S_train[1,3] == 0.014630732733543539
    # @test S_train[3,5] == -10.101157440166308
    # @test S_val[1,3] == -11.836979129448117
    # @test S_val[3,5] == -7.3722899992196025

    L = JudiLing.make_combined_L_matrix(
      latin_train,
      latin_val,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      ncol=5)

    S_train, S_val = JudiLing.make_combined_S_matrix(
      latin_train,
      latin_val,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      L)

    # @test L.L[1,3] == 0.5842161243366263
    # @test L.L[3,5] == -3.2550740303710355
    # @test L.i2f[3] == "p1"
    # @test L.f2i["p1"] == 3
    # @test S_train[1,3] == 0.014630732733543539
    # @test S_train[3,5] == -10.101157440166308
    # @test S_val[1,3] == -11.836979129448117
    # @test S_val[3,5] == -7.3722899992196025
    @test true
  catch
    @test false
  end
end

@testset "make L matrix" begin
  try
    data = DataFrame(
      X = ["A", "B", "C", "D", "E"], 
      Y = ["M", "F", "F", "M", missing], 
      Z = ["P", "P", "S", missing, "P"])

    n_features = 3
    seed = 314

    L1 = JudiLing.make_L_matrix(
      data[1:5,:],
      ["Y"],
      ["Z"],
      ncol=n_features,
      seed=seed,
      isdeep=true)

    @test size(L1.L) == (4,3)
    @test L1.f2i == Dict("S" => 4,"M" => 1,"P" => 3,"F" => 2)
    @test L1.i2f == ["M", "F", "P", "S"]
    @test L1.ncol == 3

    L2 = JudiLing.make_L_matrix(
      data[1:5,:],
      ["Y"],
      ncol=n_features,
      seed=seed,
      isdeep=true)

    @test size(L2.L) == (2,3)
    @test L2.f2i == Dict("M" => 1,"F" => 2)
    @test L2.i2f == ["M", "F"]
    @test L2.ncol == 3

    L3 = JudiLing.make_L_matrix(
      data[1:5,:],
      ["Y"],
      ["Z"],
      ncol=n_features,
      seed=seed,
      isdeep=false)

    @test size(L3.L) == (4,3)
    @test L3.f2i == Dict("S" => 4,"M" => 1,"P" => 3,"F" => 2)
    @test L3.i2f == ["M", "F", "P", "S"]
    @test L3.ncol == 3

    L4 = JudiLing.make_L_matrix(
      data[1:5,:],
      ["Y"],
      ncol=n_features,
      seed=seed,
      isdeep=false)

    @test size(L4.L) == (2,3)
    @test L4.f2i == Dict("M" => 1,"F" => 2)
    @test L4.i2f == ["M", "F"]
    @test L4.ncol == 3

    L5 = JudiLing.make_combined_L_matrix(
      data[1:2,:],
      data[3:5,:],
      ["Y"],
      ["Z"],
      ncol=n_features,
      seed=seed,
      isdeep=false)

    @test size(L5.L) == (4,3)
    @test L5.f2i == Dict("S" => 4,"M" => 1,"P" => 3,"F" => 2)
    @test L5.i2f == ["M", "F", "P", "S"]
    @test L5.ncol == 3

    L6 = JudiLing.make_combined_L_matrix(
      data[1:2,:],
      data[3:5,:],
      ["Y"],
      ncol=n_features,
      seed=seed,
      isdeep=true)

    @test size(L6.L) == (2,3)
    @test L6.f2i == Dict("M" => 1,"F" => 2)
    @test L6.i2f == ["M", "F"]
    @test L6.ncol == 3
  catch
    @test false
  end
end