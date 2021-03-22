```@meta
CurrentModule = JudiLing
```

# Preprocess

```@docs
    SplitDataException
    lpo_cv_split(p, data_path)
    loo_cv_split(data_path)
    train_val_random_split(data_path, output_dir_path, data_prefix)
    train_val_careful_split(data_path, output_dir_path, data_prefix, n_features_columns)
```