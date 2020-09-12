using JudiLing # our package
using CSV # read csv files into dataframes

# load latin file
latin = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin.csv")))
display(latin)

# create C matrixes for training and validation datasets
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

# we calculate learn_paths and hua function
res_learn = JudiLing.learn_paths(
  latin,
  latin,
  cue_obj.C,
  S,
  F,
  Chat,
  A,
  cue_obj.i2f,
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

# Once you are done, you may want to clean up the workspace
path = joinpath(@__DIR__, "latin_out")
rm(path, force=true, recursive=true)