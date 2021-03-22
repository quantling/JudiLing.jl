# Test Combo Introduction

We implemented a high-level wrapper function that aims to provide quick and preliminary studies on multiple datasets with different parameter settings. For a sophisticated study, we suggest to build a script step by step.

## Split mode
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

`:random_split` will randomly split data into training and validation datasets. In this case, it is high likely that unseen n-grams and features are in the validation datasets. Therefore, `if_combined` should be turned on. `data_path` is the path to the directory containing CSV files and `data_output_dir` is the directory for store training and validation datasets for future analysis.

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

`:careful_split` will carefully split data into training and validation datasets where there will be no unseen n-grams and features in the validation datasets. Therefore, `if_combined` shall be truned off. `data_path` is the path to the directory containing CSV files and `data_output_dir` is the directory for store training and validation datasets for future analysis. `n_features_columns` gives names of feature columns and target column.

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

## Training and validation size
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

## Make cue matrix
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

## Make S matrix
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

## Learning mode
Currently `test_combo` function supports two learning mode by `learn_mode`, `:wh` for increamental learning implemented Widrow-Hoff learning rules and `:cholesky` for end-state learning using Cholesky Decomposition.

### Cholesky
Parameters for Cholesky mode are:
- `method::Symbol = :additive`: whether :additive or :multiplicative decomposition is required
- `shift::Float64 = 0.02`: shift value for :additive decomposition
- `multiplier::Float64 = 1.01`: multiplier value for :multiplicative decomposition
- `output_format::Symbol = :auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `sparse_ratio::Float64 = 0.2`: the ratio to decide whether a matrix is sparse

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
    sparse_ratio = 0.2,
    verbose = true
    )
```

### Widrow-Hoff learning
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

### Adjacency matrix
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

### `learn_paths`
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

### `build_paths`
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

### Output directory
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