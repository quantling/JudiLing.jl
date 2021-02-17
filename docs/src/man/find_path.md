```@meta
CurrentModule = JudiLing
```

# Find Paths

```@docs
  Result_Path_Info_Struct
  Gold_Path_Info_Struct
  learn_paths(data_train, data_val, C_train, S_val, F_train, Chat_val, A, i2f, f2i)
  build_paths(data_val, C_train, S_val, F_train, Chat_val, A, i2f, C_train_ind)
  eval_can(candidates, S, F, i2f, max_can, if_pca, pca_eval_M)
  find_top_feature_indices(rC, C_train_ind)
```