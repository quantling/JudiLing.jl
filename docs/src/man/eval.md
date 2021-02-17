```@meta
CurrentModule = JudiLing
```

# Find Paths

```@docs
    Comp_Acc_Struct
    eval_SC
    eval_SC_loose
    accuracy_comprehension(S, Shat, data)
    eval_SC(SChat, SC)
    eval_SC(SChat, SC, data, target_col)
    eval_SC(SChat, SC, batch_size)
    eval_SC(SChat, SC, data, target_col, batch_size)
    eval_SC_loose(SChat, SC, k)
    eval_SC_loose(SChat, SC, k, data, target_col)
    eval_manual(res, data, i2f)
    eval_acc(res, gold_inds)
    eval_acc_loose(res, gold_inds)
    extract_gpi(gpi, threshold=0.1, tolerance=(-1000.0))
```