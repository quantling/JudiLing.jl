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

@testset "make semantic matrix" begin
  try
    data = DataFrame(
      X = ["A", "B", "C", "D", "E"],
      Y = ["M", "F", "F", "M", missing], 
      Z = ["P", "P", "S", missing, "P"])

    n_features = 3
    seed = 314

    L = JudiLing.make_L_matrix(
      data[1:5,:],
      ["Y"],
      ["Z"],
      ncol=n_features,
      seed=seed,
      isdeep=true)

    S_train, S_val = JudiLing.make_S_matrix(
      data[1:2,:],
      data[3:5,:],
      ["Y"],
      ["Z"],
      L,
      add_noise=false)

    @test S_train[1,:] == L.L[L.f2i["M"],:] + L.L[L.f2i["P"],:]
    @test S_train[2,:] == L.L[L.f2i["F"],:] + L.L[L.f2i["P"],:]
    @test S_val[1,:] == L.L[L.f2i["F"],:] + L.L[L.f2i["S"],:]
    @test S_val[2,:] == L.L[L.f2i["M"],:]
    @test S_val[3,:] == L.L[L.f2i["P"],:]

    S_train, S_val = JudiLing.make_S_matrix(
      data[1:2,:],
      data[3:5,:],
      ["Y"],
      L,
      add_noise=false)

    @test S_train[1,:] == L.L[L.f2i["M"],:]
    @test S_train[2,:] == L.L[L.f2i["F"],:]
    @test S_val[1,:] == L.L[L.f2i["F"],:]
    @test S_val[2,:] == L.L[L.f2i["M"],:]
    @test S_val[3,:] == zeros(Float64, 3)

    S_train = JudiLing.make_S_matrix(
      data[1:5,:],
      ["Y"],
      ["Z"],
      L,
      add_noise=false)

    @test S_train[1,:] == L.L[L.f2i["M"],:] + L.L[L.f2i["P"],:]
    @test S_train[2,:] == L.L[L.f2i["F"],:] + L.L[L.f2i["P"],:]
    @test S_train[3,:] == L.L[L.f2i["F"],:] + L.L[L.f2i["S"],:]
    @test S_train[4,:] == L.L[L.f2i["M"],:]
    @test S_train[5,:] == L.L[L.f2i["P"],:]

    S_train = JudiLing.make_S_matrix(
      data[1:5,:],
      ["Y"],
      L,
      add_noise=false)

    @test S_train[1,:] == L.L[L.f2i["M"],:]
    @test S_train[2,:] == L.L[L.f2i["F"],:]
    @test S_train[3,:] == L.L[L.f2i["F"],:]
    @test S_train[4,:] == L.L[L.f2i["M"],:]
    @test S_train[5,:] == zeros(Float64, 3)

    S_train, S_val = JudiLing.make_S_matrix(
      data[1:5,:],
      data[3:5,:],
      ["Y"],
      ["Z"],
      ncol=n_features,
      seed=seed,
      isdeep=true,
      add_noise=false)

    @test S_train[1,:] == L.L[L.f2i["M"],:] + L.L[L.f2i["P"],:]
    @test S_train[2,:] == L.L[L.f2i["F"],:] + L.L[L.f2i["P"],:]
    @test S_val[1,:] == L.L[L.f2i["F"],:] + L.L[L.f2i["S"],:]
    @test S_val[2,:] == L.L[L.f2i["M"],:]
    @test S_val[3,:] == L.L[L.f2i["P"],:]

    S_train = JudiLing.make_S_matrix(
      data[1:5,:],
      ["Y"],
      ["Z"],
      ncol=n_features,
      seed=seed,
      isdeep=true,
      add_noise=false)

    S_train, S_val = JudiLing.make_S_matrix(
      data[1:5,:],
      data[3:5,:],
      ["Y"],
      ncol=n_features,
      seed=seed,
      isdeep=true,
      add_noise=false)

    S_train = JudiLing.make_S_matrix(
      data[1:5,:],
      ["Y"],
      ncol=n_features,
      seed=seed,
      isdeep=true,
      add_noise=false)

    @test true
  catch
    @test false
  end
end

@testset "make combined semantic matrix" begin
  try
    data = DataFrame(
      X = ["A", "B", "C", "D", "E"],
      Y = ["M", "F", "F", "M", missing], 
      Z = ["P", "P", "S", missing, "P"])

    n_features = 3
    seed = 314

    L = JudiLing.make_L_matrix(
      data[1:5,:],
      ["Y"],
      ["Z"],
      ncol=n_features,
      seed=seed,
      isdeep=true)

    S_train, S_val = JudiLing.make_combined_S_matrix(
      data[1:2,:],
      data[3:5,:],
      ["Y"],
      ["Z"],
      ncol=n_features,
      seed=seed,
      isdeep=true,
      add_noise=false)

    @test S_train[1,:] == L.L[L.f2i["M"],:] + L.L[L.f2i["P"],:]
    @test S_train[2,:] == L.L[L.f2i["F"],:] + L.L[L.f2i["P"],:]
    @test S_val[1,:] == L.L[L.f2i["F"],:] + L.L[L.f2i["S"],:]
    @test S_val[2,:] == L.L[L.f2i["M"],:]
    @test S_val[3,:] == L.L[L.f2i["P"],:]

    S_train, S_val = JudiLing.make_combined_S_matrix(
      data[1:2,:],
      data[3:5,:],
      ["Y"],
      ncol=n_features,
      seed=seed,
      isdeep=true,
      add_noise=false)

    @test S_train[1,:] == L.L[L.f2i["M"],:]
    @test S_train[2,:] == L.L[L.f2i["F"],:]
    @test S_val[1,:] == L.L[L.f2i["F"],:]
    @test S_val[2,:] == L.L[L.f2i["M"],:]
    @test S_val[3,:] == zeros(Float64, n_features)
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