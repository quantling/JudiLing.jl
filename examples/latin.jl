using JudiLing # our package
using CSV # read csv files into dataframes

mkpath(joinpath(@__DIR__, "data"))
download("https://osf.io/2ejfu/download", "latin.csv")
mv("latin.csv", joinpath(@__DIR__, "data", "latin.csv"), force=true)
# load latin file
latin = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin.csv")))
display(latin)

# create C matrixes for training datasets
cue_obj = JudiLing.make_cue_matrix(
  latin,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

# retrieve dim of C
# we set the S matrixes as the same dimensions
n_features = size(cue_obj.C, 2)
S = JudiLing.make_S_matrix(
  latin,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  ncol=n_features)

# we use cholesky function to calculate mapping G from S to C
G = JudiLing.make_transform_matrix(S, cue_obj.C)

# we calculate Chat matrixes by multiplying S and G 
Chat = S * G

@show JudiLing.eval_SC(cue_obj.C, Chat)

# we calculate F as we did for G
F = JudiLing.make_transform_matrix(cue_obj.C, S)

# we calculate Shat as we did for Chat
Shat = cue_obj.C * F

@show JudiLing.eval_SC(S, Shat)

# here we only use a adjacency matrix as we got it from training dataset
A = cue_obj.A

# we calculate how many timestep we need for learn_paths and huo function
max_t = JudiLing.cal_max_timestep(latin, :Word)

# we calculate learn_paths and build_paths function
res_learn, gpi_learn = JudiLing.learn_paths(
  latin,
  latin,
  cue_obj.C,
  S,
  F,
  Chat,
  A,
  cue_obj.i2f,
  check_gold_path=true,
  gold_ind=cue_obj.gold_ind,
  Shat_val=Shat,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.1,
  tokenized=false,
  keep_sep=false,
  target_col=:Word,
  verbose=true)

acc_learn = JudiLing.eval_acc(
  res_learn,
  cue_obj.gold_ind,
  verbose=false
)

println("Acc for learn: $acc_learn")

res_build = JudiLing.build_paths(
    latin,
    cue_obj.C,
    S,
    F,
    Chat,
    A,
    cue_obj.i2f,
    cue_obj.gold_ind,
    max_t=max_t,
    n_neighbors=3,
    verbose=true
    )

acc_build = JudiLing.eval_acc(
  res_build,
  cue_obj.gold_ind,
  verbose=false
)

println("Acc for build: $acc_build")

# you can save results into csv files or dfs
JudiLing.write2csv(
  res_learn,
  latin,
  cue_obj,
  cue_obj,
  "latin_learn_res.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="latin_out"
  )

df_learn = JudiLing.write2df(
  res_learn,
  latin,
  cue_obj,
  cue_obj,
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word
  )

JudiLing.write2csv(
  res_build,
  latin,
  cue_obj,
  cue_obj,
  "latin_build_res.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="latin_out"
  )

df_build = JudiLing.write2df(
  res_build,
  latin,
  cue_obj,
  cue_obj,
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word
  )

display(df_learn)
display(df_build)

download("https://osf.io/2ejfu/download", "latin_train.csv")
download("https://osf.io/bm7y6/download", "latin_val.csv")
mv("latin_train.csv", joinpath(@__DIR__, "data", "latin_train.csv"), force=true)
mv("latin_val.csv", joinpath(@__DIR__, "data", "latin_val.csv"), force=true)
# cross-validation
latin_train = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin_train.csv")))
latin_val = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin_val.csv")))

# create C matrices for both training and validation datasets
cue_obj_train, cue_obj_val = JudiLing.make_cue_matrix(
  latin_train,
  latin_val,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

# create S matrices
n_features = size(cue_obj_train.C, 2)
S_train, S_val = JudiLing.make_S_matrix(
  latin_train,
  latin_val,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  ncol=n_features)

# here we learning mapping only from training dataset
G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)
F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

# we predict S and C for both training and validation datasets
Chat_train = S_train * G_train
Chat_val = S_val * G_train
Shat_train = cue_obj_train.C * F_train
Shat_val = cue_obj_val.C * F_train

# we evaluate them
@show JudiLing.eval_SC(Chat_train, cue_obj_train.C)
@show JudiLing.eval_SC(Chat_val, cue_obj_val.C)
@show JudiLing.eval_SC(Shat_train, S_train)
@show JudiLing.eval_SC(Shat_val, S_val)

# we can use build path and learn path
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
  verbose=true)

res_val, gpi_val = JudiLing.learn_paths(
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
  tolerance=-0.1,
  max_tolerance=2,
  tokenized=false,
  sep_token="-",
  keep_sep=false,
  target_col=:Word,
  issparse=:dense,
  verbose=true)

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

@show acc_train
@show acc_val

res_train = JudiLing.build_paths(
  latin_train,
  cue_obj_train.C,
  S_train,
  F_train,
  Chat_train,
  A,
  cue_obj_train.i2f,
  cue_obj_train.gold_ind,
  max_t=max_t,
  n_neighbors=3,
  verbose=true
  )

res_val = JudiLing.build_paths(
  latin_val,
  cue_obj_train.C,
  S_val,
  F_train,
  Chat_val,
  A,
  cue_obj_train.i2f,
  cue_obj_train.gold_ind,
  max_t=max_t,
  n_neighbors=20,
  verbose=true
  )

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

@show acc_train
@show acc_val

# However, we also have a wrapper function containing all above functionalities plus cross-validation into one function
JudiLing.test_combo(
  joinpath("data", "latin.csv"),
  joinpath("latin_out"),
  ["Lexeme","Person","Number","Tense","Voice","Mood"],
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  n_grams_target_col=:Word,
  grams=3,
  path_method=:learn_paths,
  train_threshold=0.1,
  val_threshold=0.01,
  csv_dir="latin_out",
  verbose=true)

# Once you are done, you may want to clean up the workspace
path = joinpath(@__DIR__, "latin_out")
rm(path, force=true, recursive=true)