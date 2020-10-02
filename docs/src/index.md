# JudiLing

## Installation

JudiLing can be installed using the Julia package manager via GitHub web links. In Julia 1.4 REPL, we can run:
```
julia> Pkg.add(PackageSpec(url="https://github.com/MegamindHenry/JudiLing.jl.git"))
```
and in Julia 1.5 REPL, we can run:
```
julia> Pkg.add(url="https://github.com/MegamindHenry/JudiLing.jl.git")
```
Or from the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add https://github.com/MegamindHenry/JudiLing.jl.git
```

## Running Julia with multiple threads
JudiLing supports the use of multiple threads. Simply start up Julia in your terminal as follows:

```
$ julia -t your_num_of_threads
```

For detailed information on using Julia with threads, see this [link](https://docs.julialang.org/en/v1/manual/multi-threading/).

## Include packages
Before we start, we first need to include two packages in julia:

```julia
using JudiLing # our package
using CSV # read csv files into dataframes
```

## Quick start example
The Latin dataset [latin.csv](https://osf.io/2ejfu/download) contains lexemes and inflectional features for 672 inflected Latin verb forms for 8 lexemes from 4 conjugation classes. Word forms are inflected for person, number, tense, voice and mood.

```
"","Word","Lexeme","Person","Number","Tense","Voice","Mood"
"1","vocoo","vocare","p1","sg","present","active","ind"
"2","vocaas","vocare","p2","sg","present","active","ind"
"3","vocat","vocare","p3","sg","present","active","ind"
"4","vocaamus","vocare","p1","pl","present","active","ind"
"5","vocaatis","vocare","p2","pl","present","active","ind"
"6","vocant","vocare","p3","pl","present","active","ind"
```

We first download and read the csv file into Julia:

```julia
download("https://osf.io/2ejfu/download", "latin.csv")

latin = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "latin.csv")));
```

and we can take a peek at the latin dataframe:
```julia
display(latin)
```
```
672×8 DataFrame. Omitted printing of 2 columns
│ Row │ Column1 │ Word           │ Lexeme  │ Person │ Number │ Tense      │
│     │ Int64   │ String         │ String  │ String │ String │ String     │
├─────┼─────────┼────────────────┼─────────┼────────┼────────┼────────────┤
│ 1   │ 1       │ vocoo          │ vocare  │ p1     │ sg     │ present    │
│ 2   │ 2       │ vocaas         │ vocare  │ p2     │ sg     │ present    │
│ 3   │ 3       │ vocat          │ vocare  │ p3     │ sg     │ present    │
│ 4   │ 4       │ vocaamus       │ vocare  │ p1     │ pl     │ present    │
│ 5   │ 5       │ vocaatis       │ vocare  │ p2     │ pl     │ present    │
│ 6   │ 6       │ vocant         │ vocare  │ p3     │ pl     │ present    │
│ 7   │ 7       │ clamoo         │ clamare │ p1     │ sg     │ present    │
│ 8   │ 8       │ clamaas        │ clamare │ p2     │ sg     │ present    │
⋮
│ 664 │ 664     │ carpsisseemus  │ carpere │ p1     │ pl     │ pluperfect │
│ 665 │ 665     │ carpsisseetis  │ carpere │ p2     │ pl     │ pluperfect │
│ 666 │ 666     │ carpsissent    │ carpere │ p3     │ pl     │ pluperfect │
│ 667 │ 667     │ cuccurissem    │ currere │ p1     │ sg     │ pluperfect │
│ 668 │ 668     │ cuccurissees   │ currere │ p2     │ sg     │ pluperfect │
│ 669 │ 669     │ cuccurisset    │ currere │ p3     │ sg     │ pluperfect │
│ 670 │ 670     │ cuccurisseemus │ currere │ p1     │ pl     │ pluperfect │
│ 671 │ 671     │ cuccurisseetis │ currere │ p2     │ pl     │ pluperfect │
│ 672 │ 672     │ cuccurissent   │ currere │ p3     │ pl     │ pluperfect │
```

For the production model, we want to predict correct forms given their lexemes and inflectional features. For example, giving the lexeme `vocare` and its inflectional features `p1`, `sg`, `present`, `active` and `ind`, the model should produce the form `vocoo`. On the other hand, the comprehension model takes forms as input and tries to predict their lexemes and inflectional features.

We use letter trigrams to encode our forms. For word `vocoo`, for example, we use trigrams `#vo`, `voc`, `oco`, `coo` and `oo#`. Here, `#` is used as start/end token to encode the initial trigram and finial trigram of a word. The row vectors of the C matrix specify for each word which of the trigrams are realized in that word.

To make the C matrix, we use the make\_cue\_matrix function:

```julia
cue_obj = JudiLing.make_cue_matrix(
  latin,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )
```

Next, we simulate the semantic matrix S using the make\_S\_matrix function:
```julia
n_features = size(cue_obj.C, 2)
S = JudiLing.make_S_matrix(
  latin,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  ncol=n_features)
```
For this simulation, first random vectors are assigned to every lexeme and inflectional feature, and next the vectors of those features are summed up to obtain the semantic vector of the inflected form. Similar dimensions for C and S work best. Therefore, we retrieve the number of columns from the C matrix and pass it to make\_S\_Matrix when constructing S.

Then, the next step is to calculate a mapping from S to C by solving equation C = SG. We use Cholesky decomposition to solve this equation:

```julia
G = JudiLing.make_transform_matrix(S, cue_obj.C)
```

Then, we can make our predicted C matrix Chat:
```julia
Chat = S * G
```

and evaluate the model's prediction accuracy:
```julia
@show JudiLing.eval_SC(cue_obj.C, Chat)
```
```
JudiLing.eval_SC(cue_obj.C, Chat) = 1.0
```

Similar to G and Chat, we can solve S = CF:
```julia
F = JudiLing.make_transform_matrix(cue_obj.C, S)
```
and we then calculate the Shat matrix and evaluate comprehension accuracy:
```julia
Shat = cue_obj.C * F
@show JudiLing.eval_SC(S, Shat)
```
```
JudiLing.eval_SC(S, Shat) = 1.0
```
To model speech production, the proper triphones have to be selected and put into the right order. We have two algorithms that accomplish this. Both algorithms construct paths in a triphone space that start with word-initial triphones and end with word-final triphones.

The first step is to construct an adjacency matrix that specify which triphone can follow each other. In this example, we use the adjacency matrix constructed by make\_cue\_matrix, but we can also make use of a independently constructed adjacency matrix if required.

```julia
A = cue_obj.A
```

For our sequencing algorithms, we calculate the number of timesteps we need for our algorithms. For the Latin dataset, the max timestep is equal to the length of the longest word. The argument :Word specifies the column in the Latin dataset that lists the words' forms.

```julia
max_t = JudiLing.cal_max_timestep(latin, :Word)
```

One sequence finding algorithm used discrimination learning for the position of triphones. This function returns two lists, one with the triphone paths and one with the learning supports for these paths.

```julia
res, gpi = JudiLing.learn_paths(
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
```

We evaluate the accuracy on the training data as follows:

```julia
acc = JudiLing.eval_acc(
  res,
  cue_obj.gold_ind,
  verbose=false
)

println("Acc for train: $acc")
```
```
Acc for train: 1.0
```

The second sequence finding algorithm is usually faster than the first, but does not
provide learnability estimates.

```julia
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
```
```
Acc for build: 1.0
```

After having obtained the results from the sequence functions: learn_paths or build_paths, we can save the results either into a csv or into a dataframe, the dataframe can be loaded into R with the rput command of the RCall package.

```julia
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
```
```
280×9 DataFrame. Omitted printing of 7 columns
│ Row │ utterance │ identifier │
│     │ Int64?    │ String?    │
├─────┼───────────┼────────────┤
│ 1   │ 1         │ vocoo      │
│ 2   │ 2         │ vocaas     │
│ 3   │ 2         │ vocaas     │
│ 4   │ 2         │ vocaas     │
│ 5   │ 3         │ vocat      │
│ 6   │ 4         │ vocaamus   │
│ 7   │ 4         │ vocaamus   │
│ 8   │ 5         │ vocaatis   │
⋮
│ 272 │ 196       │ vocaamur   │
│ 273 │ 197       │ vocaaminii │
│ 274 │ 197       │ vocaaminii │
│ 275 │ 198       │ vocantur   │
│ 276 │ 198       │ vocantur   │
│ 277 │ 199       │ clamor     │
│ 278 │ 200       │ clamaaris  │
│ 279 │ 200       │ clamaaris  │
│ 280 │ 200       │ clamaaris  │
671×9 DataFrame. Omitted printing of 7 columns
│ Row │ utterance │ identifier │
│     │ Int64?    │ String?    │
├─────┼───────────┼────────────┤
│ 1   │ 1         │ vocoo      │
│ 2   │ 1         │ vocoo      │
│ 3   │ 1         │ vocoo      │
│ 4   │ 2         │ vocaas     │
│ 5   │ 2         │ vocaas     │
│ 6   │ 2         │ vocaas     │
│ 7   │ 2         │ vocaas     │
│ 8   │ 3         │ vocat      │
⋮
│ 663 │ 198       │ vocantur   │
│ 664 │ 198       │ vocantur   │
│ 665 │ 199       │ clamor     │
│ 666 │ 199       │ clamor     │
│ 667 │ 199       │ clamor     │
│ 668 │ 200       │ clamaaris  │
│ 669 │ 200       │ clamaaris  │
│ 670 │ 200       │ clamaaris  │
│ 671 │ 200       │ clamaaris  │
```
The model also provides functionality for cross-validation. Here, you can download our datasets, [latin_train.csv](https://osf.io/yr9a3/download) and [latin_val.csv](https://osf.io/bm7y6/download). Please notice that currently our model only support validation datasets that have all n-grams seen in the training datasets.

```
download("https://osf.io/2ejfu/download", "latin_train.csv")
download("https://osf.io/bm7y6/download", "latin_val.csv")

latin_train = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "latin_train.csv")))
latin_val = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "latin_val.csv")))
```

Then, we make C matrices and S matrices.
```
cue_obj_train, cue_obj_val = JudiLing.make_cue_matrix(
  latin_train,
  latin_val,
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
```

After that, we make transformation matrices, but this time we only use training dataset and use these matrices to predict validation dataset.
```
G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)
F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

Chat_train = S_train * G_train
Chat_val = S_val * G_train
Shat_train = cue_obj_train.C * F_train
Shat_val = cue_obj_val.C * F_train

@show JudiLing.eval_SC(Chat_train, cue_obj_train.C)
@show JudiLing.eval_SC(Chat_val, cue_obj_val.C)
@show JudiLing.eval_SC(Shat_train, S_train)
@show JudiLing.eval_SC(Shat_val, S_val)
```

Then, we can find possible paths through `build_paths` or `learn_paths`.

```
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
```

However, we also have a wrapper function containing all above functionalities. You can quickly explore multiple datasets with a few optimizations.

```julia
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
```

Once you are done, you may want to clean up the workspace:
```julia
path = joinpath(@__DIR__, "latin_out")
rm(path, force=true, recursive=true)
```

You can download and try out this script [here](https://osf.io/sa89x/download).

## Citation

If you find our work helpful, please cite these following papers:

- Baayen, R. H., Chuang, Y. Y., Shafaei-Bajestan, E., and Blevins, J. P. (2019). The discriminative lexicon: A unified computational model for the lexicon and lexical processing in comprehension and production grounded not in (de)composition but in linear discriminative learning. Complexity, 2019, 1-39.

- Baayen, R. H., Chuang, Y. Y., and Blevins, J. P. (2018). Inflectional morphology with linear mappings. The Mental Lexicon, 13 (2), 232-270.

- Chuang, Y.-Y., Lõo, K., Blevins, J. P., and Baayen, R. H. (in press). Estonian case inflection made simple. A case study in Word and Paradigm morphology with Linear Discriminative Learning. In Körtvélyessy, L., and Štekauer, P. (Eds.) Complex Words: Advances in Morphology, 1-19.

- Chuang, Y-Y., Bell, M. J., Banke, I., and Baayen, R. H. (accepted). Bilingual and multilingual mental lexicon: a modeling study with Linear Discriminative Learning. Language Learning, 1-55.