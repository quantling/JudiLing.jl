```@meta
CurrentModule = JudiLing
```

# Utils

```@docs
    wh_learn(
        X,
        Y;
        eta = 0.01,
        n_epochs = 1,
        weights = nothing,
        learn_seq = nothing,
        save_history = false,
        history_cols = nothing,
        history_rows = nothing,
        verbose = false,
        )
    make_learn_seq(freq; random_seed = 314)
```
