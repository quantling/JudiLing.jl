using JuLDL
using Test

@testset "JuLDL.jl" begin
    @test JuLDL.test_func(1,0) == 1
    @test JuLDL.test_func(11,10) == 21
end
