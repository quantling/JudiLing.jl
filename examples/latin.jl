using JudiLing # our package
using CSV # read csv files into dataframes

# load latin file
latin_train = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin.csv")))
# simulate val data
# real dataset split function will be released later
latin_val = latin_train[101:150,:]

# create C matrixes for training and validation datasets
cue_obj_train = JudiLing.make_cue_matrix(
  latin_train,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )
cue_obj_val = JudiLing.make_cue_matrix(
  latin_val,
  cue_obj_train,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

# retrieve dim of C
# we set the S matrixes as the same dimensions
n_features = size(cue_obj_train.C, 2)
S_train, S_val = JudiLing.make_S_matrix(
  latin_train,
  latin_val,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  ncol=n_features)

# we use cholesky function to calculate mapping G from S to C
G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)

# we calculate Chat matrixes by multiplying S and G 
Chat_train = S_train * G_train
Chat_val = S_val * G_train

@show JudiLing.eval_SC(cue_obj_train.C, Chat_train)
@show JudiLing.eval_SC(cue_obj_val.C, Chat_val)

# we calculate F as we did for G
F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

# we calculate Shat as we did for Chat
Shat_train = cue_obj_train.C * F_train
Shat_val = cue_obj_val.C * F_train

@show JudiLing.eval_SC(S_train, Shat_train)
@show JudiLing.eval_SC(S_val, Shat_val)

# here we only use a adjacency matrix as we got it from training dataset
A = cue_obj_train.A

# we calculate how many timestep we need for learn_paths and huo function
max_t = JudiLing.cal_max_timestep(latin_train, latin_val, :Word)

# we calculate learn_paths and hua function
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

println("Acc for train: $acc_train")
println("Acc for val: $acc_val")