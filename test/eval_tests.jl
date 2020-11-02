using JudiLing
using CSV
using Test

@testset "eval for latin" begin
  try
    latin_train = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
    cue_obj_train = JudiLing.make_cue_matrix(
      latin_train,
      grams=3,
      target_col=:Word,
      tokenized=false,
      keep_sep=false
      )

    latin_val = latin_train[101:150,:]
    cue_obj_val = JudiLing.make_cue_matrix(
      latin_val,
      cue_obj_train,
      grams=3,
      target_col=:Word,
      tokenized=false,
      keep_sep=false
      )

    n_features = size(cue_obj_train.C, 2)

    S_train, S_val = JudiLing.make_S_matrix(
      latin_train,
      latin_val,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      ncol=n_features)

    G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)

    Chat_train = S_train * G_train
    Chat_val = S_val * G_train
    JudiLing.eval_SC(cue_obj_train.C, Chat_train)
    JudiLing.eval_SC(cue_obj_val.C, Chat_val)

    F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

    Shat_train = cue_obj_train.C * F_train
    Shat_val = cue_obj_val.C * F_train
    JudiLing.eval_SC(S_train, Shat_train)
    JudiLing.eval_SC(S_val, Shat_val)

    JudiLing.accuracy_comprehension(
      S_train,
      Shat_train,
      latin_train,
      target_col=:Word,
      base=["Lexeme"],
      inflections=["Person","Number","Tense","Voice","Mood"]
      )

    JudiLing.accuracy_comprehension(
      S_val,
      Shat_val,
      latin_val,
      target_col=:Word,
      base=["Lexeme"],
      inflections=["Person","Number","Tense","Voice","Mood"]
      )

    A = cue_obj_train.A

    max_t = JudiLing.cal_max_timestep(latin_train, latin_val, :Word)

    res_train, gpi_train = JudiLing.learn_paths(
      latin_train,
      latin_train,
      cue_obj_train.C,
      S_train,
      F_train,
      Chat_train,
      A,
      cue_obj_train.i2f,
      cue_obj_train.f2i,
      gold_ind=cue_obj_train.gold_ind,
      Shat_val=Shat_train,
      check_gold_path=true,
      max_t=max_t,
      max_can=10,
      grams=3,
      threshold=0.1,
      tokenized=false,
      sep_token="_",
      keep_sep=false,
      target_col=:Word,
      issparse=:dense,
      verbose=false)

    res_val, gpi_val = JudiLing.learn_paths(
      latin_train,
      latin_val,
      cue_obj_train.C,
      S_val,
      F_train,
      Chat_val,
      A,
      cue_obj_train.i2f,
      cue_obj_train.f2i,
      gold_ind=cue_obj_val.gold_ind,
      Shat_val=Shat_val,
      check_gold_path=true,
      max_t=max_t,
      max_can=10,
      grams=3,
      threshold=0.1,
      is_tolerant=true,
      tolerance=0.1,
      max_tolerance=0,
      tokenized=false,
      sep_token="-",
      keep_sep=false,
      target_col=:Word,
      issparse=:dense,
      verbose=false)

    acc_train = JudiLing.eval_acc(
      res_train,
      cue_obj_train.gold_ind,
      verbose=false
    )
    acc_val = JudiLing.eval_acc(
      res_val,
      cue_obj_val.gold_ind,
      verbose=false
    )
    acc_train_loose = JudiLing.eval_acc_loose(
      res_train,
      cue_obj_train.gold_ind,
      verbose=false
    )
    acc_val_loose = JudiLing.eval_acc_loose(
      res_val,
      cue_obj_val.gold_ind,
      verbose=false
    )

    @test true
  catch e
    @test false
  end
end

@testset "eval_SC" begin
  try
    A = [
    1 0 0
    0 1 0
    0 0 1
    1 0 0
    0 1 0
    0 0 1
    1 0 0
    0 1 0
    0 0 1
    0 0 1
    ]

    B = [
    1 0 0
    0 1 0
    0 0 1
    1 0 0
    1 0 0
    0 0 1
    1 0 0
    0 1 0
    0 0 1
    0 0 1
    ]

    C = [
    1 0 0
    0 1 0
    0 0 1
    1 0 0
    1 0 0
    1 0 0
    1 0 0
    1 0 0
    1 0 0
    0 0 1
    ]

    @test JudiLing.eval_SC(A,B) == 0.9
    @test JudiLing.eval_SC(A,C) == 0.6
    @test JudiLing.eval_SC(C,B) == 0.7
    @test JudiLing.eval_SC(A,B,2) == 0.9
    @test JudiLing.eval_SC(A,B,3) == 0.9
    @test JudiLing.eval_SC(A,B,4) == 0.9
    @test JudiLing.eval_SC(A,B,4) == 0.9
    @test JudiLing.eval_SC(A,C,2) == 0.6
    @test JudiLing.eval_SC(C,B,2) == 0.7
  catch
    @test false
  end
end