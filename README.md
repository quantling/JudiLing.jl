# JudiLing

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://quantling.github.io/JudiLing.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://quantling.github.io/JudiLing.jl/dev)
[![Build Status](https://github.com/quantling/JudiLing.jl/workflows/CI/badge.svg)](https://github.com/quantling/JudiLing.jl/actions)
[![codecov](https://codecov.io/gh/MegamindHenry/JudiLing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MegamindHenry/JudiLing.jl)

JudiLing: An implementation for Linear Discriminative Learning in Julia

Maintainer: Maria Heitmeier [@MariaHei](https://github.com/MariaHei)\
Original codebase: Xuefeng Luo [@MegamindHenry](https://github.com/MegamindHenry)

## Installation

```
using Pkg
Pkg.add("JudiLing")
```

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
Note that Flux and PyCall have to be installed separately, and the newest version of Flux requires at least Julia 1.9. If you want to run deep learning in a GPU, make sure to also install and import [CUDA](https://cuda.juliagpu.org/stable/).

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

## Supports

There are two types of supports in outputs. An utterance level and a set of supports for each cue. The former support is also called "synthesis-by-analysis" support. This support is calculated by predicted S vector and original S vector and it is used to select the best paths. Cue level supports are slices of Yt matrices from each timestep. Those supports are used to determine whether a cue is eligible for constructing paths.

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
