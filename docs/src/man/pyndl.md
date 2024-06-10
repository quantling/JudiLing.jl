```@meta
CurrentModule = JudiLing
```

JudiLing is able to call the python package [pyndl](https://github.com/quantling/pyndl) internally to compute NDL models. pyndl uses event files to compute the mapping matrices, which have to be generated manually or by using pyndl in Python, see documentation [here](https://pyndl.readthedocs.io/en/latest/#creating-grapheme-clusters-from-corpus-data).
The advantage of calling pyndl from JudiLing is that the resulting weights, cue and semantic matrices can be directly translated into JudiLing format and further processing can be done in JudiLing.

!!! note
    For pyndl to be available in JudiLing, PyCall has to be imported before JudiLing:
    ```julia
    using PyCall
    using JudiLing
    ```

## Calling pyndl from JudiLing

```@docs
    Pyndl_Weight_Struct
    pyndl(
        data_path::String;
        alpha::Float64 = 0.1,
        betas::Tuple{Float64,Float64} = (0.1, 0.1),
        method::String = "openmp"
    )
```

## Translating output of pyndl to cue and semantic matrices in JudiLing

With the weights in hand, the cue and semantic matrices can be computed:

```@docs
    make_cue_matrix(
        data::DataFrame,
        pyndl_weights::Pyndl_Weight_Struct;
        grams = 3,
        target_col = "Words",
        tokenized = false,
        sep_token = nothing,
        keep_sep = false,
        start_end_token = "#",
        verbose = false,
    )
    make_S_matrix(
        data::DataFrame,
        pyndl_weights::Pyndl_Weight_Struct,
        n_features_columns::Vector;
        tokenized::Bool=false,
        sep_token::String="_"
    )
    make_S_matrix(
        data_train::DataFrame,
        data_val::DataFrame,
        pyndl_weights::Pyndl_Weight_Struct,
        n_features_columns::Vector;
        tokenized::Bool=false,
        sep_token::String="_"
    )
```
