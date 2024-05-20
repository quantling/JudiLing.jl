using Test
using PyCall
using JudiLing
using CSV, DataFrames

using Conda
Conda.pip_interop(true)
Conda.pip("install -U","pyndl")

train = DataFrame(CSV.File(joinpath("data", "careful", "latin_train.csv")))
val = DataFrame(CSV.File(joinpath("data", "careful", "latin_val.csv")))

cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(train, val, target_col="Word", grams=3)
# S_train, S_val = JudiLing.make_combined_S_matrix(train, val, ["Lexeme"],
# ["Person", "Number", "Tense", "Voice", "Mood"])

@testset "basic" begin

    weights_latin = JudiLing.pyndl("data/latin_train_events.tab.gz",
                  alpha=0.1, betas=(0.1, 0.1), method="threading")

    cue_obj_train = JudiLing.make_cue_matrix(
    train,
    weights_latin;
    grams = 3,
    target_col = "Word",
    verbose=true
    );

    cue_obj_val = JudiLing.make_cue_matrix(
    val,
    weights_latin;
    grams = 3,
    target_col = "Word"
)   ;

    @test length(weights_latin.cues) == length(cue_obj_train.i2f)

    @test all(weights_latin.cues .== [cue_obj_train.i2f[i] for i in 1:length(weights_latin.cues)])
    @test all([cue_obj_val.i2f[i] .== ngram for (i, ngram) in enumerate(weights_latin.cues)])

    feature_columns = ["Lexeme", "Person", "Number", "Tense", "Voice", "Mood"]

    unique_features = unique(vcat(eachcol(train[:,feature_columns])...))

    S_train, S_val = JudiLing.make_S_matrix(train, val,
    weights_latin,
    feature_columns,
    tokenized=false)

    @test size(S_train,2) == length(unique_features)
    @test size(S_val,2) == length(unique_features)

    @test all((vec(sum(S_train, dims=2))) .== length(feature_columns))

    @test all(S_train[:, weights_latin.outcomes .== "currere"] .== (train.Lexeme .== "currere"))
    @test all(S_train[:, weights_latin.outcomes .== "p3"] .== (train.Person .== "p3"))

    S_train = JudiLing.make_S_matrix(train,
    weights_latin,
    feature_columns,
    tokenized=false)

    S_val = JudiLing.make_S_matrix(val,
    weights_latin,
    feature_columns,
    tokenized=false)

    @test size(S_train,2) == length(unique_features)
    @test size(S_val,2) == length(unique_features)

    @test all((vec(sum(S_train, dims=2))) .== length(feature_columns))

    @test all(S_train[:, weights_latin.outcomes .== "currere"] .== (train.Lexeme .== "currere"))
    @test all(S_train[:, weights_latin.outcomes .== "p3"] .== (train.Person .== "p3"))
end
