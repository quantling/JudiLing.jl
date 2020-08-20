using JuLDL
using Test

@testset "JuLDL.jl" begin
  @test JuLDL.test_func(1,2) == 4
  @test JuLDL.test_func(4,2) == 10
end
