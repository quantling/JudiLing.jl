# JudiLing

!!! note
    If you encounter an error like "ERROR: UndefVarError: DataFrame! not defined", this is because our dependency CSV.jl changed their APIs in v0.8. Please use "data = DataFrame(CSV.File(path_to_csv_file))" to read a CSV file and include DataFrames package by "using DataFrames".

## Installation

You can install JudiLing by the follow commands:
```
using Pkg
Pkg.add("JudiLing")
```
For brave adventurers, install test version of JudiLing by:
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
Before we start, we first need to load the JudiLing package:

```julia
using JudiLing
```

**Note:**
As of JudiLing 0.8.0, [PyCall](https://github.com/JuliaPy/PyCall.jl) and [Flux](https://fluxml.ai/Flux.jl/stable/) have become optional dependencies. This means that all code in JudiLing which requires calls to python is only available if PyCall is loaded first, like this:
```julia
using PyCall
using JudiLing
```
Likewise, the code involving deep learning is only available if Julia's deep learning library Flux is loaded first, like this:
```julia
using Flux
using JudiLing
```
Note that Flux and PyCall have to be installed separately, and the newest version of Flux requires at least Julia 1.9.

## Running Julia with multiple threads
JudiLing supports the use of multiple threads. Simply start up Julia in your terminal as follows:

```
$ julia -t your_num_of_threads
```

For detailed information on using Julia with threads, see this [link](https://docs.julialang.org/en/v1/manual/multi-threading/).

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

latin = JudiLing.load_dataset("latin.csv");
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

To make the C matrix, we use the `make_cue_matrix` function:

```julia
cue_obj = JudiLing.make_cue_matrix(
    latin,
    grams=3,
    target_col=:Word,
    tokenized=false,
    keep_sep=false
    )
```

Next, we simulate the semantic matrix S using the `make_S_matrix` function:
```julia
n_features = size(cue_obj.C, 2)
S = JudiLing.make_S_matrix(
    latin,
    ["Lexeme"],
    ["Person","Number","Tense","Voice","Mood"],
    ncol=n_features)
```
For this simulation, first random vectors are assigned to every lexeme and inflectional feature, and next the vectors of those features are summed up to obtain the semantic vector of the inflected form. Similar dimensions for C and S work best. Therefore, we retrieve the number of columns from the C matrix and pass it to `make_S_matrix` when constructing S.

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
@show JudiLing.eval_SC(Chat, cue_obj.C)
```

Output:
```output
JudiLing.eval_SC(Chat, cue_obj.C) = 0.9926
```

> **_NOTE:_** Accuracy may be different depending on the simulated semantic matrix.

Similar to G and Chat, we can solve S = CF:
```julia
F = JudiLing.make_transform_matrix(cue_obj.C, S)
```
and we then calculate the Shat matrix and evaluate comprehension accuracy:
```julia
Shat = cue_obj.C * F
@show JudiLing.eval_SC(Shat, S)
```

Output:
```output
JudiLing.eval_SC(Shat, S) = 0.9911
```
> **_NOTE:_** Accuracy may be different depending on the simulated semantic matrix.

To model speech production, the proper triphones have to be selected and put into the right order. We have two algorithms that accomplish this. Both algorithms construct paths in a triphone space that start with word-initial triphones and end with word-final triphones.

The first step is to construct an adjacency matrix that specify which triphone can follow each other. In this example, we use the adjacency matrix constructed by `make_cue_matrix`, but we can also make use of a independently constructed adjacency matrix if required.

```julia
A = cue_obj.A
```

For our sequencing algorithms, we calculate the number of timesteps we need for our algorithms. For the Latin dataset, the max timestep is equal to the length of the longest word. The argument `:Word` specifies the column in the Latin dataset that lists the words' forms.

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

## Cross-validation

The model also provides functionality for cross-validation. Here, we first split the dataset randomly into 90% training and 10% validation data:

```
latin_train, latin_val = JudiLing.loading_data_randomly_split("latin.csv",
                                                                "data",
                                                                "latin",
                                                                val_ratio=0.1,
                                                                random_seed=42)
```

Then, we make the C matrix by passing both training and validation datasets to the `make_combined_cue_matrix` function which ensures that the C matrix contains columns for
both training and validation data.
```
cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(
    latin_train,
    latin_val,
    grams = 3,
    target_col = :Word,
    tokenized = false,
    keep_sep = false
)
```

Next, we simulate semantic vectors, again for both the training and validation data,
using `make_combined_S_matrix`:
```
n_features = size(cue_obj_train.C, 2)
S_train, S_val = JudiLing.make_combined_S_matrix(
    latin_train,
    latin_val,
    ["Lexeme"],
    ["Person", "Number", "Tense", "Voice", "Mood"],
    ncol = n_features
)
```

After that, we make the transformation matrices, but this time we only use the training dataset. We use these transformation matrices to predict the validation dataset.
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
JudiLing.eval_SC(Chat_train, cue_obj_train.C) = 0.995
JudiLing.eval_SC(Chat_val, cue_obj_val.C) = 0.403
JudiLing.eval_SC(Shat_train, S_train) = 0.9917
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
acc_learn_train = 0.9983
acc_learn_val = 0.6866
acc_build_train = 1.0
acc_build_val = 0.3284
```

Alternatively, we  have a wrapper function incorporating all above functionalities. With this function, you can quickly explore datasets with different parameter settings. Please find more in the [Test Combo Introduction](@ref).

## Test Combo Introduction

We implemented a high-level wrapper function that aims to provide quick and preliminary studies on multiple datasets with different parameter settings. For a sophisticated study, we suggest to build a script step by step.

In general, `test_combo` function will perform the following operations:

- prepare datasets
- make cue matrix object
- make semantic matrix
- learn transfrom mapping F and G
- perform path-finding algorithms for both `learn_paths` and `build_paths` in training and validation datasets
- evaluate results
- save outputs

You can download the available datasets you need for the following demos. ([french.csv](https://osf.io/b3mju/download), [estonian_train.csv](https://osf.io/3xvp4/download) and [estonian_val.csv](https://osf.io/zqt2c/download))

### Split mode
`test_combo` function provides four split mode. `:train_only` give the opportunity to only evaluate the model with training data or partial training data. `data_path` is the path to the CSV file and `data_output_dir` is the directory for store training and validation datasets for future analysis.

```julia
JudiLing.test_combo(
    :train_only,
    data_path = joinpath(@__DIR__, "data", "latin.csv"),
    data_prefix = "latin",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_grams_target_col = :Word,
    n_grams_tokenized = false,
    grams = 3,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
    verbose = true
    )
```

`:pre_split` give the option for datasets that already have been split into train and validation datasets. `data_path` is the path to the directory containing CSV files.

```julia
JudiLing.test_combo(
    :pre_split,
    data_path=joinpath(@__DIR__, "data"),
    data_prefix="estonian",
    n_grams_target_col=:Word,
    n_grams_tokenized=false,
    grams=3,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Lexeme","Case","Number"],
    A_mode = :train_only,
    threshold_train = 0.1,
    is_tolerant_train = false,
    threshold_val = 0.1,
    is_tolerant_val = true,
    tolerance_val = -0.1,
    max_tolerance_val = 3,
    verbose = true
    )
```

`:random_split` will randomly split data into training and validation datasets. In this case, it is high likely that unseen n-grams and features are in the validation datasets. Therefore, you should set `if_combined` to true. `data_path` is the path to the directory containing CSV files and `data_output_dir` is the directory for store training and validation datasets for future analysis.

```julia
JudiLing.test_combo(
    :random_split,
    val_sample_size = 1000,
    data_path = joinpath(@__DIR__, "data", "french.csv"),
    data_prefix = "french",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_grams_target_col = :Syllables,
    n_grams_tokenized = true,
    n_grams_sep_token = "-",
    n_grams_keep_sep = true,
    grams = 2,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
    if_combined = true,
    threshold_train = 0.1,
    is_tolerant_train = false,
    is_tolerant_val = true,
    threshold_val = 0.1,
    tolerance_val = -0.1,
    max_tolerance_train = 3,
    verbose = true
    )
```

`:careful_split` will carefully split data into training and validation datasets where there will be no unseen n-grams and features in the validation datasets. Therefore, you should set `if_combined` to false. `data_path` is the path to the directory containing CSV files and `data_output_dir` is the directory for store training and validation datasets for future analysis. `n_features_columns` gives names of feature columns and target column.

```julia
JudiLing.test_combo(
    :careful_split,
    val_sample_size = 1000,
    data_path = joinpath(@__DIR__, "data", "french.csv"),
    data_prefix = "french",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_features_columns = ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
    n_grams_target_col = :Syllables,
    n_grams_tokenized = true,
    n_grams_sep_token = "-",
    n_grams_keep_sep = true,
    grams = 2,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
    if_combined = true,
    threshold_train = 0.1,
    is_tolerant_train = false,
    is_tolerant_val = true,
    threshold_val = 0.1,
    tolerance_val = -0.1,
    max_tolerance_train = 3,
    verbose = true
    )
```

### Training and validation size
`val_sample_size` and `val_ratio` control the validation data size. `train_sample_size` controls the training data size. For very large datasets, it is recommended that try out with small `train_sample_size` first, then test out the whole dataset.

```julia
JudiLing.test_combo(
    :random_split,
    train_sample_size = 3000,
    val_sample_size = 100,
    data_path = joinpath(@__DIR__, "data", "french.csv"),
    data_prefix = "french",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_features_columns = ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
    n_grams_target_col = :Syllables,
    n_grams_tokenized = true,
    n_grams_sep_token = "-",
    n_grams_keep_sep = true,
    grams = 2,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
    if_combined = true,
    threshold_train = 0.1,
    is_tolerant_train = false,
    is_tolerant_val = true,
    threshold_val = 0.1,
    tolerance_val = -0.1,
    max_tolerance_train = 3,
    verbose = true
    )

JudiLing.test_combo(
    :random_split,
    val_ratio = 0.1,
    data_path = joinpath(@__DIR__, "data", "french.csv"),
    data_prefix = "french",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_features_columns = ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
    n_grams_target_col = :Syllables,
    n_grams_tokenized = true,
    n_grams_sep_token = "-",
    n_grams_keep_sep = true,
    grams = 2,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
    if_combined = true,
    threshold_train = 0.1,
    is_tolerant_train = false,
    is_tolerant_val = true,
    threshold_val = 0.1,
    tolerance_val = -0.1,
    max_tolerance_train = 3,
    verbose = true
    )
```

### Make cue matrix
Parameters for making cue matrix object is same as `make_cue_matrix` function:
- `grams::Int64=3`: the number of grams for cues
- `n_grams_target_col::Union{String, Symbol}=:Words`: the column name for target strings
- `n_grams_tokenized::Bool=false`:if true, the dataset target is assumed to be tokenized
- `n_grams_sep_token::Union{Nothing, String, Char}=nothing`: separator
- `n_grams_keep_sep::Bool=false`: if true, keep separators in cues
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues

```julia
JudiLing.test_combo(
    :train_only,
    data_path = joinpath(@__DIR__, "data", "latin.csv"),
    data_prefix = "latin",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_features_columns = ["Lexeme","Person","Number","Tense","Voice","Mood"],
    n_grams_target_col = :Word,
    n_grams_tokenized = false,
    n_grams_sep_token = nothing,
    n_grams_keep_sep = false,
    grams = 3,
    start_end_token = "#",
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
    verbose = true
    )

JudiLing.test_combo(
    :random_split,
    val_sample_size = 1000,
    data_path = joinpath(@__DIR__, "data", "french.csv"),
    data_prefix = "french",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_features_columns = ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
    n_grams_target_col = :Syllables,
    n_grams_tokenized = true,
    n_grams_sep_token = "-",
    n_grams_keep_sep = true,
    grams = 2,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
    if_combined = true,
    threshold_train = 0.1,
    is_tolerant_train = false,
    is_tolerant_val = true,
    threshold_val = 0.1,
    tolerance_val = -0.1,
    max_tolerance_train = 3,
    verbose = true
    )
```

### Make S matrix
Parameters for making S matrix is the same as `make_S_matrix`:
- `n_features_base::Vector`: context lexemes
- `n_features_inflections::Vector`: grammatic lexemes
- `sd_base_mean::Int64=1`: the sd mean of base features
- `sd_inflection_mean::Int64=1`: the sd mean of inflectional features
- `sd_base::Int64=4`: the sd of base features
- `sd_inflection::Int64=4`: the sd of inflectional features
- `isdeep::Bool=true`: if true, mean of each feature is also randomized
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd

```julia
JudiLing.test_combo(
    :train_only,
    data_path = joinpath(@__DIR__, "data", "latin.csv"),
    data_prefix = "latin",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_grams_target_col = :Word,
    n_grams_tokenized = false,
    grams = 3,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
    sd_base_mean = 1,
    sd_inflection_mean = 1,
    sd_base = 4,
    sd_inflection = 4,
    isdeep = true,
    add_noise = true,
    sd_noise = 1,
    normalized = false,
    verbose = true
    )
```

### Learning mode
Currently `test_combo` function supports two learning mode by `learn_mode`, `:wh` for increamental learning implemented Widrow-Hoff learning rules and `:cholesky` for end-state learning using Cholesky Decomposition.

#### Cholesky
Parameters for Cholesky mode are:
- `method::Symbol = :additive`: whether :additive or :multiplicative decomposition is required
- `shift::Float64 = 0.02`: shift value for :additive decomposition
- `multiplier::Float64 = 1.01`: multiplier value for :multiplicative decomposition
- `output_format::Symbol = :auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `sparse_ratio::Float64 = 0.05`: the ratio to decide whether a matrix is sparse

```julia
JudiLing.test_combo(
    :train_only,
    data_path = joinpath(@__DIR__, "data", "latin.csv"),
    data_prefix = "latin",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_grams_target_col = :Word,
    n_grams_tokenized = false,
    grams = 3,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
    learn_mode = :cholesky,
    method = :additive,
    shift = 0.02,
    output_format = :auto,
    sparse_ratio = 0.05,
    verbose = true
    )
```

#### Widrow-Hoff learning
Parameters for Widrow-Hoff learning are:
- `wh_freq::Vector = nothing`: the learning sequence
- `init_weights::Matrix = nothing`: the initial weights
- `eta::Float64 = 0.1`: the learning rate
- `n_epochs::Int64 = 1`: the number of epochs to be trained

```julia
JudiLing.test_combo(
    :train_only,
    data_path = joinpath(@__DIR__, "data", "latin.csv"),
    data_prefix = "latin",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_grams_target_col = :Word,
    n_grams_tokenized = false,
    grams = 3,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
    learn_mode = :wh,
    eta = 0.001,
    n_epochs = 1000,
    verbose = true
    )
```

#### Adjacency matrix
`test_combo` has control (`A_mode`) for whether to take combined adjacency matrix (`:combined`). In that case, the adjacency matrix is made from both training and validation matrix, otherwise the adjacency matrix is only made from training data (`:train_only`). There is also an option to pass custumized adjacency matrix (`A`).

```julia
JudiLing.test_combo(
    :random_split,
    val_sample_size = 1000,
    data_path = joinpath(@__DIR__, "data", "french.csv"),
    data_prefix = "french",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_features_columns = ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
    n_grams_target_col = :Syllables,
    n_grams_tokenized = true,
    n_grams_sep_token = "-",
    n_grams_keep_sep = true,
    grams = 2,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
    if_combined = true,
    A_mode = :combined,
    threshold_train = 0.1,
    is_tolerant_train = false,
    is_tolerant_val = true,
    threshold_val = 0.1,
    tolerance_val = -0.1,
    max_tolerance_train = 3,
    verbose = true
    )

# suppose we had A matrix from somewhere else
JudiLing.test_combo(
    :random_split,
    val_sample_size = 1000,
    data_path = joinpath(@__DIR__, "data", "french.csv"),
    data_prefix = "french",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_features_columns = ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
    n_grams_target_col = :Syllables,
    n_grams_tokenized = true,
    n_grams_sep_token = "-",
    n_grams_keep_sep = true,
    grams = 2,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
    if_combined = true,
    A = A,
    threshold_train = 0.1,
    is_tolerant_train = false,
    is_tolerant_val = true,
    threshold_val = 0.1,
    tolerance_val = -0.1,
    max_tolerance_train = 3,
    verbose = true
    )
```

#### `learn_paths`
We have separate parameters for training and validation data:
- `threshold_train::Float64 = 0.1`: the value set for the support such that if the support of an n-gram is higher than this value, the n-gram will be taking into consideration
- `is_tolerant_train::Bool = false`: if true, select a specified number (given by `max_tolerance`) of n-grams whose supports are below threshold but above a second tolerance threshold to be added to the path
- `tolerance_train::Float64 = -0.1`: the value set for the second threshold (in tolerant mode) such that if the support for an n-gram is in between this value and the threshold and the max_tolerance number has not been reached, then allow this n-gram to be added to the path
- `max_tolerance_train::Int64 = 2`: maximum number of n-grams allowed in a path
- `threshold_val::Float64 = 0.1`: the value set for the support such that if the support of an n-gram is higher than this value, the n-gram will be taking into consideration
- `is_tolerant_val::Bool = false`: if true, select a specified number (given by `max_tolerance`) of n-grams whose supports are below threshold but above a second tolerance threshold to be added to the path
- `tolerance_val::Float64 = -0.1`: the value set for the second threshold (in tolerant mode) such that if the support for an n-gram is in between this value and the threshold and the max_tolerance number has not been reached, then allow this n-gram to be added to the path
- `max_tolerance_val::Int64 = 2`: maximum number of n-grams allowed in a path

```julia
JudiLing.test_combo(
    :random_split,
    train_sample_size = 3000,
    val_sample_size = 100,
    data_path = joinpath(@__DIR__, "data", "french.csv"),
    data_prefix = "french",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_features_columns = ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
    n_grams_target_col = :Syllables,
    n_grams_tokenized = true,
    n_grams_sep_token = "-",
    n_grams_keep_sep = true,
    grams = 2,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
    if_combined = true,
    threshold_train = 0.1,
    is_tolerant_train = false,
    is_tolerant_val = true,
    threshold_val = 0.1,
    tolerance_val = -0.1,
    max_tolerance_train = 3,
    verbose = true
    )
```

#### `build_paths`
We have separate parameters for training and validation data:
- `n_neighbors_train::Int64 = 10`: the top n form neighbors to be considered
- `n_neighbors_val::Int64 = 20`: the top n form neighbors to be considered

```julia
JudiLing.test_combo(
    :random_split,
    train_sample_size = 3000,
    val_sample_size = 100,
    data_path = joinpath(@__DIR__, "data", "french.csv"),
    data_prefix = "french",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_features_columns = ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
    n_grams_target_col = :Syllables,
    n_grams_tokenized = true,
    n_grams_sep_token = "-",
    n_grams_keep_sep = true,
    grams = 2,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
    if_combined = true,
    n_neighbors_train = 10,
    n_neighbors_val = 20,
    verbose = true
    )
```

#### Output directory
All outputs will be stored in a directory which can be configured by `output_dir`.

```julia
JudiLing.test_combo(
    :train_only,
    data_path = joinpath(@__DIR__, "data", "latin.csv"),
    data_prefix = "latin",
    data_output_dir = joinpath(@__DIR__, "data"),
    n_grams_target_col = :Word,
    n_grams_tokenized = false,
    grams = 3,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
    output_dir = joinpath(@__DIR__, "latin_out"),
    verbose = true
    )
```

## Acknowledgments

This project was supported by the ERC advanced grant WIDE-742545 and by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC number 2064/1 - Project number 390727645.

## Citation

If you find this package helpful, please cite it as follows:

Luo, X., Heitmeier, M., Chuang, Y. Y., Baayen, R. H. JudiLing: an implementation of the Discriminative Lexicon Model in Julia. Eberhard Karls Universität Tübingen, Seminar für Sprachwissenschaft.

The following studies have made use of several algorithms now implemented in JudiLing instead of WpmWithLdl:

- Baayen, R. H., Chuang, Y. Y., Shafaei-Bajestan, E., and Blevins, J. P. (2019). The discriminative lexicon: A unified computational model for the lexicon and lexical processing in comprehension and production grounded not in (de)composition but in linear discriminative learning. Complexity, 2019, 1-39.

- Baayen, R. H., Chuang, Y. Y., and Blevins, J. P. (2018). Inflectional morphology with linear mappings. The Mental Lexicon, 13 (2), 232-270.

- Chuang, Y.-Y., Lõo, K., Blevins, J. P., and Baayen, R. H. (2020). Estonian case inflection made simple. A case study in Word and Paradigm morphology with Linear Discriminative Learning. In Körtvélyessy, L., and Štekauer, P. (Eds.) Complex Words: Advances in Morphology, 1-19.

- Chuang, Y-Y., Bell, M. J., Banke, I., and Baayen, R. H. (2020). Bilingual and multilingual mental lexicon: a modeling study with Linear Discriminative Learning. Language Learning, 1-55.

- Heitmeier, M., Chuang, Y-Y., Baayen, R. H. (2021). Modeling morphology with Linear Discriminative Learning: considerations and design choices. Frontiers in Psychology, 12, 4929.

- Denistia, K., and Baayen, R. H. (2022). The morphology of Indonesian: Data and quantitative modeling. In Shei, C., and Li, S. (Eds.) The Routledge Handbook of Asian Linguistics, (pp. 605-634). Routledge, London.

- Heitmeier, M., Chuang, Y.-Y., and Baayen, R. H. (2023). How trial-to-trial learning shapes mappings in the mental lexicon: Modelling lexical decision with linear discriminative learning. Cognitive Psychology, 1-30.

- Chuang, Y. Y., Kang, M., Luo, X. F. and Baayen, R. H. (2023). Vector Space Morphology with Linear Discriminative Learning. In Crepaldi, D. (Ed.) Linguistic morphology in the mind and brain.

- Heitmeier, M., Chuang, Y. Y., Axen, S. D., & Baayen, R. H. (2024). Frequency effects in linear discriminative learning. Frontiers in Human Neuroscience, 17, 1242720.

- Plag, I., Heitmeier, M. & Domahs, F. (to appear). German nominal number interpretation in an impaired mental lexicon: A naive discriminative learning perspective. The Mental Lexicon.
