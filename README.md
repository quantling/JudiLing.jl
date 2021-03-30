# JudiLing

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MegamindHenry.github.io/JudiLing.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MegamindHenry.github.io/JudiLing.jl/dev)
[![Build Status](https://github.com/MegamindHenry/JudiLing.jl/workflows/CI/badge.svg)](https://github.com/MegamindHenry/JudiLing.jl/actions)
[![codecov](https://codecov.io/gh/MegamindHenry/JudiLing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MegamindHenry/JudiLing.jl)

JudiLing: An implementation for Linear Discriminative Learning in Julia

**Note:** If you encounter an error like "ERROR: UndefVarError: DataFrame! not defined", this is because our dependency CSV.jl changed their APIs in v0.8. Please use "data = DataFrame(CSV.File(path_to_csv_file))" to read a CSV file and include DataFrames package by "using DataFrames".

## Installation

JudiLing is now on the Julia package system. You can install JudiLing by the follow commands:
```
using Pkg
Pkg.add("JudiLing")
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
using DataFrames # parse data into dataframes
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

latin = DataFrame(CSV.File(joinpath(@__DIR__, "latin.csv")));
```

and we can inspect the latin dataframe:
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

Output:
```output
JudiLing.eval_SC(Chat, cue_obj.C) = 0.9926
```

!!! note
    Accuracy may be different depending on the simulated semantic matrix.

Similar to G and Chat, we can solve S = CF:
```julia
F = JudiLing.make_transform_matrix(cue_obj.C, S)
```
and we then calculate the Shat matrix and evaluate comprehension accuracy:
```julia
Shat = cue_obj.C * F
@show JudiLing.eval_SC(S, Shat)
```

Output:
```output
JudiLing.eval_SC(Shat, S) = 0.9911
```

!!! note
    Accuracy may be different depending on the simulated semantic matrix.

To model speech production, the proper triphones have to be selected and put into the right order. We have two algorithms that accomplish this. Both algorithms construct paths in a triphone space that start with word-initial triphones and end with word-final triphones.

The first step is to construct an adjacency matrix that specify which triphone can follow each other. In this example, we use the adjacency matrix constructed by make\_cue\_matrix, but we can also make use of a independently constructed adjacency matrix if required.

```julia
A = cue_obj.A
```

For our sequencing algorithms, we calculate the number of timesteps we need for our algorithms. For the Latin dataset, the max timestep is equal to the length of the longest word. The argument :Word specifies the column in the Latin dataset that lists the words' forms.

```julia
max_t = JudiLing.cal_max_timestep(latin, :Word)
```

One sequence finding algorithm used discrimination learning for the position of triphones. This function returns two lists, one with candidate triphone paths and their positional learning support (res) and one with the semantic supports for the gold paths (gpi).

```julia
res_learn, gpi_learn = JudiLing.learn_paths(
    latin,
    latin,
    cue_obj.C,
    S,
    F,
    Chat,
    A,
    cue_obj.i2f,
    cue_obj.f2i, # api changed in 0.3.1
    check_gold_path = true,
    gold_ind = cue_obj.gold_ind,
    Shat_val = Shat,
    max_t = max_t,
    max_can = 10,
    grams = 3,
    threshold = 0.05,
    tokenized = false,
    keep_sep = false,
    target_col = :Word,
    verbose = true
)
```

We evaluate the accuracy on the training data as follows:

```julia
acc_learn = JudiLing.eval_acc(res_learn, cue_obj.gold_ind, verbose = false)

println("Acc for learn: $acc_learn")
```
```
Acc for learn: 0.9985
```

The second sequence finding algorithm is usually faster than the first, but does not provide positional learnability estimates.

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
Acc for build: 0.9955
```

After having obtained the results from the sequence functions: `learn_paths` or `build_paths`, we can save the results either into a csv or into a dataframe, the dataframe can be loaded into R with the rput command of the RCall package.

```julia
JudiLing.write2csv(
    res_learn,
    latin,
    cue_obj,
    cue_obj,
    "latin_learn_res.csv",
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :Word,
    root_dir = @__DIR__,
    output_dir = "latin_out"
)

df_learn = JudiLing.write2df(
    res_learn,
    latin,
    cue_obj,
    cue_obj,
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :Word
)

JudiLing.write2csv(
    res_build,
    latin,
    cue_obj,
    cue_obj,
    "latin_build_res.csv",
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :Word,
    root_dir = @__DIR__,
    output_dir = "latin_out"
)

df_build = JudiLing.write2df(
    res_build,
    latin,
    cue_obj,
    cue_obj,
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :Word
)

display(df_learn)
display(df_build)
```
```
3805×9 DataFrame. Omitted printing of 5 columns
│ Row  │ utterance │ identifier     │ path                                                    │ pred           │
│      │ Int64?    │ String?        │ Union{Missing, String}                                  │ String?        │
├──────┼───────────┼────────────────┼─────────────────────────────────────────────────────────┼────────────────┤
│ 1    │ 1         │ vocoo          │ #vo:voc:oco:coo:oo#                                     │ vocoo          │
│ 2    │ 2         │ vocaas         │ #vo:voc:oca:caa:aas:as#                                 │ vocaas         │
│ 3    │ 2         │ vocaas         │ #vo:voc:oca:caa:aab:aba:baa:aas:as#                     │ vocaabaas      │
│ 4    │ 2         │ vocaas         │ #vo:voc:oca:caa:aat:ati:tis:is#                         │ vocaatis       │
│ 5    │ 2         │ vocaas         │ #vo:voc:oca:caa:aav:avi:vis:ist:sti:tis:is#             │ vocaavistis    │
│ 6    │ 2         │ vocaas         │ #vo:voc:oca:caa:aam:amu:mus:us#                         │ vocaamus       │
│ 7    │ 2         │ vocaas         │ #vo:voc:oca:caa:aab:abi:bit:it#                         │ vocaabit       │
│ 8    │ 2         │ vocaas         │ #vo:voc:oca:caa:aam:amu:mur:ur#                         │ vocaamur       │
│ 9    │ 2         │ vocaas         │ #vo:voc:oca:caa:aar:are:ret:et#                         │ vocaaret       │
⋮
│ 3796 │ 671       │ cuccurisseetis │ #cu:cuc:ucc:ccu:cur:ure:ree:eet:eti:tis:is#             │ cuccureetis    │
│ 3797 │ 671       │ cuccurisseetis │ #cu:cuc:ucc:ccu:cur:uri:ris:ist:sti:tis:is#             │ cuccuristis    │
│ 3798 │ 671       │ cuccurisseetis │ #cu:cuc:ucc:ccu:cur:uri:ris:iss:sse:set:et#             │ cuccurisset    │
│ 3799 │ 671       │ cuccurisseetis │ #cu:cur:urr:rri:rim:imi:min:ini:nii:ii#                 │ curriminii     │
│ 3800 │ 672       │ cuccurissent   │ #cu:cuc:ucc:ccu:cur:uri:ris:iss:sse:sen:ent:nt#         │ cuccurissent   │
│ 3801 │ 672       │ cuccurissent   │ #cu:cur:urr:rre:rer:ere:ren:ent:nt#                     │ currerent      │
│ 3802 │ 672       │ cuccurissent   │ #cu:cuc:ucc:ccu:cur:uri:ris:iss:sse:see:eem:emu:mus:us# │ cuccurisseemus │
│ 3803 │ 672       │ cuccurissent   │ #cu:cuc:ucc:ccu:cur:uri:ris:iss:sse:see:eet:eti:tis:is# │ cuccurisseetis │
│ 3804 │ 672       │ cuccurissent   │ #cu:cur:urr:rre:rer:ere:ren:ent:ntu:tur:ur#             │ currerentur    │
│ 3805 │ 672       │ cuccurissent   │ #cu:cuc:ucc:ccu:cur:uri:ris:iss:sse:see:ees:es#         │ cuccurissees   │
2519×9 DataFrame. Omitted printing of 4 columns
│ Row  │ utterance │ identifier     │ path                                            │ pred         │ num_tolerance │
│      │ Int64?    │ String?        │ Union{Missing, String}                          │ String?      │ Int64?        │
├──────┼───────────┼────────────────┼─────────────────────────────────────────────────┼──────────────┼───────────────┤
│ 1    │ 1         │ vocoo          │ #vo:voc:oco:coo:oo#                             │ vocoo        │ 0             │
│ 2    │ 1         │ vocoo          │ #vo:voc:oca:caa:aab:abo:boo:oo#                 │ vocaaboo     │ 0             │
│ 3    │ 1         │ vocoo          │ #vo:voc:oca:caa:aab:aba:bam:am#                 │ vocaabam     │ 0             │
│ 4    │ 2         │ vocaas         │ #vo:voc:oca:caa:aas:as#                         │ vocaas       │ 0             │
│ 5    │ 2         │ vocaas         │ #vo:voc:oca:caa:aab:abi:bis:is#                 │ vocaabis     │ 0             │
│ 6    │ 2         │ vocaas         │ #vo:voc:oca:caa:aat:ati:tis:is#                 │ vocaatis     │ 0             │
│ 7    │ 3         │ vocat          │ #vo:voc:oca:cat:at#                             │ vocat        │ 0             │
│ 8    │ 3         │ vocat          │ #vo:voc:oca:caa:aab:aba:bat:at#                 │ vocaabat     │ 0             │
│ 9    │ 3         │ vocat          │ #vo:voc:oca:caa:aas:as#                         │ vocaas       │ 0             │
⋮
│ 2510 │ 671       │ cuccurisseetis │ #cu:cur:uri:ris:iss:sse:see:ees:es#             │ curissees    │ 0             │
│ 2511 │ 671       │ cuccurisseetis │ #cu:cur:uri:ris:iss:sse:see:eem:emu:mus:us#     │ curisseemus  │ 0             │
│ 2512 │ 671       │ cuccurisseetis │ #cu:cur:uri:ris:is#                             │ curis        │ 0             │
│ 2513 │ 671       │ cuccurisseetis │ #cu:cuc:ucc:ccu:cur:uri:ris:is#                 │ cuccuris     │ 0             │
│ 2514 │ 672       │ cuccurissent   │ #cu:cuc:ucc:ccu:cur:uri:ris:iss:sse:sen:ent:nt# │ cuccurissent │ 0             │
│ 2515 │ 672       │ cuccurissent   │ #cu:cur:uri:ris:iss:sse:sen:ent:nt#             │ curissent    │ 0             │
│ 2516 │ 672       │ cuccurissent   │ #cu:cuc:ucc:ccu:cur:uri:ris:iss:sse:set:et#     │ cuccurisset  │ 0             │
│ 2517 │ 672       │ cuccurissent   │ #cu:cur:uri:ris:iss:sse:set:et#                 │ curisset     │ 0             │
│ 2518 │ 672       │ cuccurissent   │ #cu:cuc:ucc:ccu:cur:uri:ris:iss:sse:sem:em#     │ cuccurissem  │ 0             │
│ 2519 │ 672       │ cuccurissent   │ #cu:cur:uri:ris:iss:sse:sem:em#                 │ curissem     │ 0             │
```
The model also provides functionality for cross-validation. Here, you can download our datasets, [latin_train.csv](https://osf.io/yr9a3/download) and [latin_val.csv](https://osf.io/bm7y6/download). Please notice that currently our model only support validation datasets that have all their n-grams present in the training datasets.

```
download("https://osf.io/2ejfu/download", joinpath(@__DIR__, "data", "latin_train.csv"))
download("https://osf.io/bm7y6/download", joinpath(@__DIR__, "data", "latin_val.csv"))

latin_train =
    DataFrame(CSV.File(joinpath(@__DIR__, "data", "latin_train.csv")))
latin_val =
    DataFrame(CSV.File(joinpath(@__DIR__, "data", "latin_val.csv")))
```

Then, we make the C and S matrices passing both training and validation datasets to the `make_cue_matrix` function.
```
cue_obj_train, cue_obj_val = JudiLing.make_cue_matrix(
    latin_train,
    latin_val,
    grams = 3,
    target_col = :Word,
    tokenized = false,
    keep_sep = false
)

n_features = size(cue_obj_train.C, 2)
S_train, S_val = JudiLing.make_S_matrix(
    latin_train,
    latin_val,
    ["Lexeme"],
    ["Person", "Number", "Tense", "Voice", "Mood"],
    ncol = n_features
)
```

After that, we make the transformation matrices, but this time we only use training dataset. We use these transformation matrices to predict the validation dataset.
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

Output:
```output
JudiLing.eval_SC(Chat_train, cue_obj_train.C) = 0.9926
JudiLing.eval_SC(Chat_val, cue_obj_val.C) = 0.3955
JudiLing.eval_SC(Shat_train, S_train) = 0.9911
JudiLing.eval_SC(Shat_val, S_val) = 1.0
```

Finally, we can find possible paths through `build_paths` or `learn_paths`. Since validation datasets are harder to predict, we turn on `tolerant` mode which allow the algorithms to find more paths but at the cost of investing more time.

```
A = cue_obj_train.A
max_t = JudiLing.cal_max_timestep(latin_train, latin_val, :Word)

res_learn_train, gpi_learn_train = JudiLing.learn_paths(
    latin_train,
    latin_train,
    cue_obj_train.C,
    S_train,
    F_train,
    Chat_train,
    A,
    cue_obj_train.i2f,
    cue_obj_train.f2i, # api changed in 0.3.1
    gold_ind = cue_obj_train.gold_ind,
    Shat_val = Shat_train,
    check_gold_path = true,
    max_t = max_t,
    max_can = 10,
    grams = 3,
    threshold = 0.05,
    tokenized = false,
    sep_token = "_",
    keep_sep = false,
    target_col = :Word,
    issparse = :dense,
    verbose = true,
)

res_learn_val, gpi_learn_val = JudiLing.learn_paths(
    latin_train,
    latin_val,
    cue_obj_train.C,
    S_val,
    F_train,
    Chat_val,
    A,
    cue_obj_train.i2f,
    cue_obj_train.f2i, # api changed in 0.3.1
    gold_ind = cue_obj_val.gold_ind,
    Shat_val = Shat_val,
    check_gold_path = true,
    max_t = max_t,
    max_can = 10,
    grams = 3,
    threshold = 0.05,
    is_tolerant = true,
    tolerance = -0.1,
    max_tolerance = 2,
    tokenized = false,
    sep_token = "-",
    keep_sep = false,
    target_col = :Word,
    issparse = :dense,
    verbose = true,
)

acc_learn_train =
    JudiLing.eval_acc(res_learn_train, cue_obj_train.gold_ind, verbose = false)
acc_learn_val = JudiLing.eval_acc(res_learn_val, cue_obj_val.gold_ind, verbose = false)

res_build_train = JudiLing.build_paths(
    latin_train,
    cue_obj_train.C,
    S_train,
    F_train,
    Chat_train,
    A,
    cue_obj_train.i2f,
    cue_obj_train.gold_ind,
    max_t = max_t,
    n_neighbors = 3,
    verbose = true,
)

res_build_val = JudiLing.build_paths(
    latin_val,
    cue_obj_train.C,
    S_val,
    F_train,
    Chat_val,
    A,
    cue_obj_train.i2f,
    cue_obj_train.gold_ind,
    max_t = max_t,
    n_neighbors = 20,
    verbose = true,
)

acc_build_train =
    JudiLing.eval_acc(res_build_train, cue_obj_train.gold_ind, verbose = false)
acc_build_val = JudiLing.eval_acc(res_build_val, cue_obj_val.gold_ind, verbose = false)

@show acc_learn_train
@show acc_learn_val
@show acc_build_train
@show acc_build_val
```

Output:
```output
acc_learn_train = 0.9985
acc_learn_val = 0.8433
acc_build_train = 0.9955
acc_build_val = 0.8433
```

Alternatively, we  have a wrapper function incorporating all above functionalities. With this function, you can quickly explore datasets with different parameter settings. Please find more in the [Test Combo Introduction](@ref).

Once you are done, you may want to clean up your output directory:

```julia
rm(joinpath(@__DIR__, "data"), force = true, recursive = true)
rm(joinpath(@__DIR__, "latin_out"), force = true, recursive = true)
```

You can download and try out this script [here](https://osf.io/sa89x/download).

## Citation

If you find this package helpful, please cite this as follow:

Luo, X., Chuang, Y. Y., Baayen, R. H. JudiLing: an implementation in Julia of Linear Discriminative Learning algorithms for language model. Eberhard Karls Universität Tübingen, Seminar für Sprachwissenschaft.

The following studies have made use of several algorithms now implemented in JudiLing instead of WpmWithLdl:

- Baayen, R. H., Chuang, Y. Y., Shafaei-Bajestan, E., and Blevins, J. P. (2019). The discriminative lexicon: A unified computational model for the lexicon and lexical processing in comprehension and production grounded not in (de)composition but in linear discriminative learning. Complexity, 2019, 1-39.

- Baayen, R. H., Chuang, Y. Y., and Blevins, J. P. (2018). Inflectional morphology with linear mappings. The Mental Lexicon, 13 (2), 232-270.

- Chuang, Y.-Y., Lõo, K., Blevins, J. P., and Baayen, R. H. (in press). Estonian case inflection made simple. A case study in Word and Paradigm morphology with Linear Discriminative Learning. In Körtvélyessy, L., and Štekauer, P. (Eds.) Complex Words: Advances in Morphology, 1-19.

- Chuang, Y-Y., Bell, M. J., Banke, I., and Baayen, R. H. (accepted). Bilingual and multilingual mental lexicon: a modeling study with Linear Discriminative Learning. Language Learning, 1-55.