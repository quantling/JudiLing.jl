using JudiLing
using Test

@testset "load dataset" begin
    data = JudiLing.load_dataset("data/latin_train.csv")

    @test size(data,1) == 3

    @test typeof(data[!, "Word"]) == Vector{String}

    data = JudiLing.load_dataset("data/latin_train.csv", limit=2)

    @test size(data,1) == 2
end
