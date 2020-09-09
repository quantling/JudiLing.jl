# JudiLing

## Installation

JudiLing can be installed using the Julia package manager via GitHub HTTPS
Links.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add https://github.com/MegamindHenry/JudiLing.jl.git
```

## Examples
### Latin
Let's look at our first real dataset example. [latin.csv](https://github.com/MegamindHenry/JudiLing.jl/blob/master/examples/data/latin.csv) contains lexemes and inflective features of Latin verbs.
```
"","Word","Lexeme","Person","Number","Tense","Voice","Mood"
"1","vocoo","vocare","p1","sg","present","active","ind"
"2","vocaas","vocare","p2","sg","present","active","ind"
"3","vocat","vocare","p3","sg","present","active","ind"
"4","vocaamus","vocare","p1","pl","present","active","ind"
"5","vocaatis","vocare","p2","pl","present","active","ind"
"6","vocant","vocare","p3","pl","present","active","ind"
```
Before we start, we first need to add two packages in julia:

```julia
using JudiLing # our package
using CSV # read csv files into dataframes
```

Then, we need to read the csv file:

```julia
latin_train = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin.csv")))
```

and here we simulate our validation csv file, which contains the same lexemes, inflective features and cues.

```julia
latin_val = latin_train[101:150,:]
```

!!! warning

    The real validation dataset should NOT contain datarows from the training dataset.

For the production model, we want to predict correct forms given their lexemes and inflective features. For example, giving the lexeme `vocare` and its inflective features `p1`, `sg`, `present`, `active` and `ind`, the model should be able to produce the form `vocoo`. On the other hand, the comprehension model takes forms as input and try to predict their lexemes and inflective features.

We use letter n-grams to encode our forms. For word `vocoo`, for example, we encode it as `#vo`, `voc`, `oco`, `coo` and `oo#`. Here, `#` is used as start/end token to encode the beginning and ending of a word. To make the C matrix, we use:

```julia
cue_obj_train = JudiLing.make_cue_matrix(
  latin_train,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )
```

In order to maintain the same indices as in the training C matrix, we need to pass cue\_obj\_train when we construct validation C matrix.

```julia
cue_obj_val = JudiLing.make_cue_matrix(
  latin_val,
  cue_obj_train,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )
```

Then, we can simulate our training and validation S matrices by:
```julia
n_features = size(cue_obj_train.C, 2)
S_train, S_val = JudiLing.make_S_matrix(
  latin_train,
  latin_val,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  ncol=n_features)
```
For this simulation, random vectors are assigned to every lexeme and inflective feature and we sum up lexemes and features as our semantic vector for each word entry. As shown from our study, similar dimensions between C and S work best. Thus, we retrieve dimension from C matrix and pass it to construct S matrix.

Then, the next step is to calculate a mapping from S to C by solving equation C = SG. We have several mapping modes, but here we use cholesky decomposition mode to solve the equation:

```julia
G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)
```

Then, we can make our predicted C matrix Chat:
```julia
Chat_train = S_train * G_train
Chat_val = S_val * G_train
```

and we can evaluate our predictions by:
```julia
@show JudiLing.eval_SC(cue_obj_train.C, Chat_train)
@show JudiLing.eval_SC(cue_obj_val.C, Chat_val)
```

Similar to G and Chat, we can solve S = CF by:
```julia
F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)
```
and we can predict Shat matrices and evaluate them:
```julia
Shat_train = cue_obj_train.C * F_train
Shat_val = cue_obj_val.C * F_train

@show JudiLing.eval_SC(S_train, Shat_train)
@show JudiLing.eval_SC(S_val, Shat_val)
```

We have path finding algorithms to further predict the forms in a sequence. The first step is to construct an adjacency matrix. Here we use adjacency constructed inside make\_cue\_matrix(), but we can also make a full adjacency matrix if we need.
```julia
A = cue_obj_train.A
```

Then, we calculate hthe number of timesteps we need for our learning function.
```julia
max_t = JudiLing.cal_max_timestep(latin_train, latin_val, :Word)
```

Finally, we use our path finding function to predict forms in a sequence.
```julia
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
```
and we evaluate our results:
```julia
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
```

After we obtain results from path finding functions like learn\_paths or build\_paths, we can later save the results into a csv file or into a dataframe which can be loaded into R.
Examples are like these:

```julia
JudiLing.write2csv(
  res_train,
  latin_train,
  cue_obj_train,
  cue_obj_train,
  "latin_train_res.csv",
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

JudiLing.write2csv(
  res_val,
  latin_val,
  cue_obj_train,
  cue_obj_val,
  "latin_val_res.csv",
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

JudiLing.write2csv(
  gpi_train,
  "latin_train_gpi.csv",
  root_dir=@__DIR__,
  output_dir="latin_out"
  )

JudiLing.write2csv(
  gpi_val,
  "latin_val_gpi.csv",
  root_dir=@__DIR__,
  output_dir="latin_out"
  )

df_train = JudiLing.write2df(
  res_train,
  latin_train,
  cue_obj_train,
  cue_obj_train,
  "latin_train_res.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word
  )

df_val = JudiLing.write2df(
  res_val,
  latin_val,
  cue_obj_train,
  cue_obj_val,
  "latin_val_res.csv",
  grams=3,
  tokenized=false,
  sep_token=nothing,
  start_end_token="#",
  output_sep_token="",
  path_sep_token=":",
  target_col=:Word
  )

display(df_train)
display(df_val)

# you may want to delete the temp out folder
path = joinpath(@__DIR__, "latin_out")
rm(path, force=true, recursive=true)
```

We can download and try our completed script [here](https://github.com/MegamindHenry/JudiLing.jl/blob/master/examples/latin.jl).