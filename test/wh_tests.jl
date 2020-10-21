using JudiLing
using Test

@testset "make learn sequence" begin
  freq = [1, 2, 3, 4]
  learn_seq = JudiLing.make_learn_seq(freq)

  @test learn_seq = [2,4,2,4,4,3,1,4,3,3]
end