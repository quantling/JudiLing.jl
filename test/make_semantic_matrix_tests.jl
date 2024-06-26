using JudiLing
using CSV
using Test
using DataFrames

@testset "make prelinguistic semantic matrix for utterance" begin

    function test_cues(features, idx, s_obj)
        some_cues = split(features[idx], "_")
        tgt_vec = zeros(length(keys(s_obj.f2i)))
        for cue in some_cues
            i = s_obj.f2i[cue]
            tgt_vec[i] = 1
        end
        @test tgt_vec == s_obj.pS[idx,:]
    end

    utterance = DataFrame(CSV.File(joinpath(
        @__DIR__,
        "data",
        "utterance_mini.csv",
    )))

    cues_split = [split(d, "_") for d in utterance.CommunicativeIntention]
    unique_cues = Set(vcat(cues_split...))

    s_obj_train = JudiLing.make_pS_matrix(utterance)

    @test length(unique_cues) == size(s_obj_train.pS, 2)
    test_cues(utterance.CommunicativeIntention, 5, s_obj_train)
    test_cues(utterance.CommunicativeIntention, 15, s_obj_train)
    test_cues(utterance.CommunicativeIntention, 23, s_obj_train)

    utterance_val = utterance[101:end, :]
    s_obj_val = JudiLing.make_pS_matrix(utterance_val, s_obj_train)
    test_cues(utterance_val.CommunicativeIntention, 5, s_obj_val)
    test_cues(utterance_val.CommunicativeIntention, 6, s_obj_val)

    @test length(unique_cues) == size(s_obj_val.pS, 2)

    s_obj_train, s_obj_val = JudiLing.make_combined_pS_matrix(utterance, utterance_val)

    @test length(unique_cues) == size(s_obj_train.pS, 2)
    test_cues(utterance.CommunicativeIntention, 5, s_obj_train)
    test_cues(utterance.CommunicativeIntention, 15, s_obj_train)
    test_cues(utterance.CommunicativeIntention, 23, s_obj_train)
    test_cues(utterance_val.CommunicativeIntention, 5, s_obj_val)
    test_cues(utterance_val.CommunicativeIntention, 6, s_obj_val)
end

@testset "make semantic matrix" begin
    data = DataFrame(
        X = ["A", "B", "C", "D", "E"],
        Y = ["M", "F", "F", "M", missing],
        Z = ["P", "P", "S", missing, "P"],
    )

    n_features = 3
    seed = 314

    L = JudiLing.make_L_matrix(
        data[1:5, :],
        ["Y"],
        ["Z"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
    )

    S_train, S_val = JudiLing.make_S_matrix(
        data[1:2, :],
        data[3:5, :],
        ["Y"],
        ["Z"],
        L,
        add_noise = false,
    )

    @test S_train[1, :] == L.L[L.f2i["M"], :] + L.L[L.f2i["P"], :]
    @test S_train[2, :] == L.L[L.f2i["F"], :] + L.L[L.f2i["P"], :]
    @test S_val[1, :] == L.L[L.f2i["F"], :] + L.L[L.f2i["S"], :]
    @test S_val[2, :] == L.L[L.f2i["M"], :]
    @test S_val[3, :] == L.L[L.f2i["P"], :]

    S_train, S_val = JudiLing.make_S_matrix(
        data[1:2, :],
        data[3:5, :],
        ["Y"],
        L,
        add_noise = false,
    )

    @test S_train[1, :] == L.L[L.f2i["M"], :]
    @test S_train[2, :] == L.L[L.f2i["F"], :]
    @test S_val[1, :] == L.L[L.f2i["F"], :]
    @test S_val[2, :] == L.L[L.f2i["M"], :]
    @test S_val[3, :] == zeros(Float64, 3)

    S_train = JudiLing.make_S_matrix(
        data[1:5, :],
        ["Y"],
        ["Z"],
        L,
        add_noise = false,
    )

    @test S_train[1, :] == L.L[L.f2i["M"], :] + L.L[L.f2i["P"], :]
    @test S_train[2, :] == L.L[L.f2i["F"], :] + L.L[L.f2i["P"], :]
    @test S_train[3, :] == L.L[L.f2i["F"], :] + L.L[L.f2i["S"], :]
    @test S_train[4, :] == L.L[L.f2i["M"], :]
    @test S_train[5, :] == L.L[L.f2i["P"], :]

    S_train =
        JudiLing.make_S_matrix(data[1:5, :], ["Y"], L, add_noise = false)

    @test S_train[1, :] == L.L[L.f2i["M"], :]
    @test S_train[2, :] == L.L[L.f2i["F"], :]
    @test S_train[3, :] == L.L[L.f2i["F"], :]
    @test S_train[4, :] == L.L[L.f2i["M"], :]
    @test S_train[5, :] == zeros(Float64, 3)

    S_train, S_val = JudiLing.make_S_matrix(
        data[1:5, :],
        data[3:5, :],
        ["Y"],
        ["Z"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
        add_noise = false,
    )

    @test S_train[1, :] == L.L[L.f2i["M"], :] + L.L[L.f2i["P"], :]
    @test S_train[2, :] == L.L[L.f2i["F"], :] + L.L[L.f2i["P"], :]
    @test S_val[1, :] == L.L[L.f2i["F"], :] + L.L[L.f2i["S"], :]
    @test S_val[2, :] == L.L[L.f2i["M"], :]
    @test S_val[3, :] == L.L[L.f2i["P"], :]

    S_train = JudiLing.make_S_matrix(
        data[1:5, :],
        ["Y"],
        ["Z"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
        add_noise = false,
    )

    S_train, S_val = JudiLing.make_S_matrix(
        data[1:5, :],
        data[3:5, :],
        ["Y"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
        add_noise = false,
    )

    S_train = JudiLing.make_S_matrix(
        data[1:5, :],
        ["Y"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
        add_noise = false,
    )
end

@testset "make combined semantic matrix" begin
    data = DataFrame(
        X = ["A", "B", "C", "D", "E"],
        Y = ["M", "F", "F", "M", missing],
        Z = ["P", "P", "S", missing, "P"],
    )

    n_features = 3
    seed = 314

    L = JudiLing.make_L_matrix(
        data[1:5, :],
        ["Y"],
        ["Z"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
    )

    S_train, S_val = JudiLing.make_combined_S_matrix(
        data[1:2, :],
        data[3:5, :],
        ["Y"],
        ["Z"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
        add_noise = false,
    )

    @test S_train[1, :] == L.L[L.f2i["M"], :] + L.L[L.f2i["P"], :]
    @test S_train[2, :] == L.L[L.f2i["F"], :] + L.L[L.f2i["P"], :]
    @test S_val[1, :] == L.L[L.f2i["F"], :] + L.L[L.f2i["S"], :]
    @test S_val[2, :] == L.L[L.f2i["M"], :]
    @test S_val[3, :] == L.L[L.f2i["P"], :]

    S_train, S_val = JudiLing.make_combined_S_matrix(
        data[1:2, :],
        data[3:5, :],
        ["Y"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
        add_noise = false,
    )

    @test S_train[1, :] == L.L[L.f2i["M"], :]
    @test S_train[2, :] == L.L[L.f2i["F"], :]
    @test S_val[1, :] == L.L[L.f2i["F"], :]
    @test S_val[2, :] == L.L[L.f2i["M"], :]
    @test S_val[3, :] == zeros(Float64, n_features)

    train = DataFrame(CSV.File(joinpath(
        @__DIR__,
        "data",
        "latin_train.csv",
    )))
    val = DataFrame(CSV.File(joinpath(
        @__DIR__,
        "data",
        "latin_val.csv",
    )))

    # check that indeed there are columns with differing string types
    println(typeof(train.Word) != typeof(val.Word))

    # but make combined cue matrix still runs
    S_train, S_val = JudiLing.make_combined_S_matrix(
        train,
        val,
        ["Lexeme"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
        add_noise = false,
    )
end

@testset "make L matrix" begin
    data = DataFrame(
        X = ["A", "B", "C", "D", "E"],
        Y = ["M", "F", "F", "M", missing],
        Z = ["P", "P", "S", missing, "P"],
    )

    n_features = 3
    seed = 314

    L1 = JudiLing.make_L_matrix(
        data[1:5, :],
        ["Y"],
        ["Z"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
    )

    @test size(L1.L) == (4, 3)
    @test L1.f2i == Dict("S" => 4, "M" => 1, "P" => 3, "F" => 2)
    @test L1.i2f == ["M", "F", "P", "S"]
    @test L1.ncol == 3

    L2 = JudiLing.make_L_matrix(
        data[1:5, :],
        ["Y"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
    )

    @test size(L2.L) == (2, 3)
    @test L2.f2i == Dict("M" => 1, "F" => 2)
    @test L2.i2f == ["M", "F"]
    @test L2.ncol == 3

    L3 = JudiLing.make_L_matrix(
        data[1:5, :],
        ["Y"],
        ["Z"],
        ncol = n_features,
        seed = seed,
        isdeep = false,
    )

    @test size(L3.L) == (4, 3)
    @test L3.f2i == Dict("S" => 4, "M" => 1, "P" => 3, "F" => 2)
    @test L3.i2f == ["M", "F", "P", "S"]
    @test L3.ncol == 3

    L4 = JudiLing.make_L_matrix(
        data[1:5, :],
        ["Y"],
        ncol = n_features,
        seed = seed,
        isdeep = false,
    )

    @test size(L4.L) == (2, 3)
    @test L4.f2i == Dict("M" => 1, "F" => 2)
    @test L4.i2f == ["M", "F"]
    @test L4.ncol == 3

    L5 = JudiLing.make_combined_L_matrix(
        data[1:2, :],
        data[3:5, :],
        ["Y"],
        ["Z"],
        ncol = n_features,
        seed = seed,
        isdeep = false,
    )

    @test size(L5.L) == (4, 3)
    @test L5.f2i == Dict("S" => 4, "M" => 1, "P" => 3, "F" => 2)
    @test L5.i2f == ["M", "F", "P", "S"]
    @test L5.ncol == 3

    L6 = JudiLing.make_combined_L_matrix(
        data[1:2, :],
        data[3:5, :],
        ["Y"],
        ncol = n_features,
        seed = seed,
        isdeep = true,
    )

    @test size(L6.L) == (2, 3)
    @test L6.f2i == Dict("M" => 1, "F" => 2)
    @test L6.i2f == ["M", "F"]
    @test L6.ncol == 3
end
