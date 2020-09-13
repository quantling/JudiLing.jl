using JudiLing # our package
using CSV # read csv files into dataframes

# load estonian file
estonian_train = CSV.DataFrame!(CSV.File(
  joinpath(@__DIR__, "data", "estonian_train.csv")))
estonian_val = CSV.DataFrame!(CSV.File(
  joinpath(@__DIR__, "data", "estonian_val.csv")))

# create C matrixes for training and validation datasets
cue_obj_train = JudiLing.make_cue_matrix(
  estonian_train,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

# in order maintain the same trigram indices, we pass training object as an argument
cue_obj_val = JudiLing.make_cue_matrix(
  estonian_val,
  cue_obj_train,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

# construct S matrices
n_features = size(cue_obj_train.C, 2)
S_train, S_val = JudiLing.make_S_matrix(
  estonian_train,
  estonian_val,
  ["Lexeme"],
  ["Lexeme","Case","Number"],
  ncol=n_features)

# we use cholesky function to calculate mapping G from S to C
# Here, we construct the G matrix only use training data
G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)

# we predict C matrices both training and validation
Chat_train = S_train * G_train
Chat_val = S_val * G_train

@show JudiLing.eval_SC(cue_obj_train.C, Chat_train)
@show JudiLing.eval_SC(cue_obj_val.C, Chat_val)

# We follow similar processes to predict and evaluate S matrices for both training and validation data as well.
F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

Shat_train = cue_obj_train.C * F_train
Shat_val = cue_obj_val.C * F_train

@show JudiLing.eval_SC(S_train, Shat_train)
@show JudiLing.eval_SC(S_val, Shat_val)

# we use adjacency matrix
A = cue_obj_train.A

# and we calculate the maximum timestep which is equal to the maximum value between training and validation data.
max_t = JudiLing.cal_max_timestep(estonian_train, estonian_val, :Word)

res_learn_train = JudiLing.learn_paths(
  estonian_train,
  estonian_train,
  cue_obj_train.C,
  S_train,
  F_train,
  Chat_train,
  A,
  cue_obj_train.i2f,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.1,
  tokenized=false,
  keep_sep=false,
  target_col=:Word,
  verbose=true)

res_learn_val = JudiLing.learn_paths(
  estonian_train,
  estonian_val,
  cue_obj_train.C,
  S_val,
  F_train,
  Chat_val,
  A,
  cue_obj_train.i2f,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.01,
  tokenized=false,
  keep_sep=false,
  target_col=:Word,
  verbose=true)

acc_learn_train = JudiLing.eval_acc(
  res_learn_train,
  cue_obj_train.gold_ind,
  verbose=false
)

acc_learn_val = JudiLing.eval_acc(
  res_learn_val,
  cue_obj_val.gold_ind,
  verbose=false
)

println("Acc for learn train: $acc_learn_train")
println("Acc for learn val: $acc_learn_val")

res_build_train = JudiLing.build_paths(
  estonian_train,
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

res_build_val = JudiLing.build_paths(
  estonian_val,
  cue_obj_train.C,
  S_val,
  F_train,
  Chat_val,
  A,
  cue_obj_train.i2f,
  cue_obj_train.gold_ind,
  max_t=max_t,
  n_neighbors=8,
  verbose=true
  )

acc_build_train = JudiLing.eval_acc(
  res_build_train,
  cue_obj_train.gold_ind,
  verbose=false
)

acc_build_val = JudiLing.eval_acc(
  res_build_val,
  cue_obj_val.gold_ind,
  verbose=false
)

println("Acc for build train: $acc_build_train")
println("Acc for build val: $acc_build_val")

#save results
JudiLing.write2csv(
  res_learn_train,
  estonian_train,
  cue_obj_train,
  cue_obj_train,
  "estonian_learn_res_train.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="estonian_out"
  )

JudiLing.write2csv(
  res_learn_val,
  estonian_val,
  cue_obj_train,
  cue_obj_val,
  "estonian_learn_res_val.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="estonian_out"
  )

JudiLing.write2csv(
  res_build_train,
  estonian_train,
  cue_obj_train,
  cue_obj_train,
  "estonian_build_res_train.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="estonian_out"
  )

JudiLing.write2csv(
  res_build_val,
  estonian_val,
  cue_obj_train,
  cue_obj_val,
  "estonian_build_res_val.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word,
  root_dir=@__DIR__,
  output_dir="estonian_out"
  )

df_learn_train = JudiLing.write2df(
  res_learn_train,
  estonian_train,
  cue_obj_train,
  cue_obj_train,
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word
  )

df_learn_val = JudiLing.write2df(
  res_learn_val,
  estonian_val,
  cue_obj_train,
  cue_obj_val,
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word
  )

df_build_train = JudiLing.write2df(
  res_build_train,
  estonian_train,
  cue_obj_train,
  cue_obj_train,
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word
  )

df_build_val = JudiLing.write2df(
  res_build_val,
  estonian_val,
  cue_obj_train,
  cue_obj_val,
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word
  )



# Once you are done, you may want to clean up the workspace
path = joinpath(@__DIR__, "estonian_out")
rm(path, force=true, recursive=true)