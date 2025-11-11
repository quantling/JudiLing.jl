using JudiLing
using Test

@testset "make learn sequence" begin
    freq = [1, 2, 3, 4]
    learn_seq = JudiLing.make_learn_seq(freq)

    @test all([count(==(i), learn_seq) == i for i in freq])

    learn_seq2 = JudiLing.make_learn_seq(freq)

    @test learn_seq == learn_seq2

    freq = [1, 2, 3, 4, 5, 6]
    learn_seq = JudiLing.make_learn_seq(freq, random_seed=42)

    @test all([count(==(i), learn_seq) == i for i in freq])
end

@testset "wh tests" begin
    C = [1 1 1 1 0 0 0 0; 1 0 1 0 1 0 1 0; 0 0 0 0 1 1 1 1]
    S = [1 0 1 0; 1 1 0 0; 0 0 1 1]

    F = JudiLing.wh_learn(
        C,
        S,
        eta = 0.001,
        n_epochs = 10000,
        )
    Shat = C * F
    @test -0.05 < sum(Shat-S) < 0.05

    G = JudiLing.wh_learn(
        S,
        C,
        eta = 0.001,
        n_epochs = 10000,
        )
    Chat = S * G
    @test -0.05 < sum(Chat-C) < 0.05
end
