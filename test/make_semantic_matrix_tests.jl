using JuLDL
using CSV
using Test

@testset "make prelinguistic semantic matrix for utterance" begin
  utterance = CSV.DataFrame!(CSV.File(joinpath("data", "utterance_mini.csv")))
  s_obj_train = JuLDL.make_pS_matrix(utterance)
  @test s_obj_train.pS[153, 172] == 1
  @test s_obj_train.pS[172,172] == 1
  @test s_obj_train.f2i["telephone"] == 107
  @test s_obj_train.i2f[75] == "does"

  utterance_val = utterance[101:end, :]
  s_obj_val = JuLDL.make_pS_matrix(utterance_val, s_obj_train)
  @test s_obj_val.pS[51, 194] == 1
  @test s_obj_val.pS[74,198] == 1
  @test s_obj_val.f2i["telephone"] == 107
  @test s_obj_val.i2f[75] == "does"
end