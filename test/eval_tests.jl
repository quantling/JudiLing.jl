using JuLDL
using CSV
using Test

@testset "eval for latin" begin
  try
    latin_train = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
    cue_obj_train = JuLDL.make_cue_matrix(
      latin_train,
      grams=3,
      words_column=:Word,
      tokenized=false,
      keep_sep=false
      )

    latin_val = latin_train[101:150,:]
    cue_obj_val = JuLDL.make_cue_matrix(
      latin_val,
      cue_obj_train,
      grams=3,
      words_column=:Word,
      tokenized=false,
      keep_sep=false
      )

    n_features = size(cue_obj_train.C, 2)

    S_train, S_val = JuLDL.make_S_matrix(
      latin_train,
      latin_val,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      ncol=n_features)

    G_train = JuLDL.make_transform_matrix(S_train, cue_obj_train.C)

    Chat_train = S_train * G_train
    Chat_val = S_val * G_train
    JuLDL.eval_SC(cue_obj_train.C, Chat_train)
    JuLDL.eval_SC(cue_obj_val.C, Chat_val)

    F_train = JuLDL.make_transform_matrix(cue_obj_train.C, S_train)

    Shat_train = cue_obj_train.C * F_train
    Shat_val = cue_obj_val.C * F_train
    JuLDL.eval_SC(S_train, Shat_train)
    JuLDL.eval_SC(S_val, Shat_val)

    A = cue_obj_train.A

    max_t = JuLDL.cal_max_timestep(latin_train, latin_val, :Word)

    res_train, gpi_train = JuLDL.shuo(
      latin_train,
      latin_train,
      cue_obj_train.C,
      S_train,
      F_train,
      Chat_train,
      A,
      cue_obj_train.i2f,
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
      words_column=:Word,
      issparse=:dense,
      verbose=false)

    res_val, gpi_val = JuLDL.shuo(
      latin_train,
      latin_val,
      cue_obj_train.C,
      S_val,
      F_train,
      Chat_val,
      A,
      cue_obj_train.i2f,
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
      words_column=:Word,
      issparse=:dense,
      verbose=false)

    acc_train = JuLDL.eval_acc(
      res_train,
      cue_obj_train.gold_ind,
      verbose=false
    )
    acc_val = JuLDL.eval_acc(
      res_val,
      cue_obj_val.gold_ind,
      verbose=false
    )
    acc_train_loose = JuLDL.eval_acc_loose(
      res_train,
      cue_obj_train.gold_ind,
      verbose=false
    )
    acc_val_loose = JuLDL.eval_acc_loose(
      res_val,
      cue_obj_val.gold_ind,
      verbose=false
    )

    @test true
  catch e
    @show e
    display(e)
    @test e == false
  end
end