```@meta
CurrentModule = JudiLing
```

# Deep learning in JudiLing

```@docs
predict_from_deep_model(model::Flux.Chain,
                                 X::Union{SparseMatrixCSC,Matrix})
predict_shat(model::Flux.Chain,
                  ci::Vector{Int})
get_and_train_model(X_train::Union{SparseMatrixCSC,Matrix},
                            Y_train::Union{SparseMatrixCSC,Matrix},
                            X_val::Union{SparseMatrixCSC,Matrix,Missing},
                            Y_val::Union{SparseMatrixCSC,Matrix,Missing},
                            data_train::Union{DataFrame,Missing},
                            data_val::Union{DataFrame,Missing},
                            target_col::Union{Symbol, String,Missing},
                            model_outpath::String;
                            hidden_dim::Int=1000,
                            n_epochs::Int=100,
                            batchsize::Int=64,
                            loss_func::Function=Flux.mse,
                            optimizer=Flux.Adam(0.001),
                            model::Union{Missing, Flux.Chain} = missing,
                            early_stopping::Union{Missing, Int}=missing,
                            optimise_for_acc::Bool=false,
                            return_losses::Bool=false,
                            verbose::Bool=true,
                            measures_func::Union{Missing, Function}=missing,
                            return_train_acc::Bool=false,
                            kargs...)
get_and_train_model(X_train::Union{SparseMatrixCSC,Matrix},
                            Y_train::Union{SparseMatrixCSC,Matrix},
                            model_outpath::String;
                            data_train::Union{Missing, DataFrame}=missing,
                            target_col::Union{Missing, Symbol, String}=missing,
                            hidden_dim::Int=1000,
                            n_epochs::Int=100,
                            batchsize::Int=64,
                            loss_func::Function=Flux.mse,
                            optimizer=Flux.Adam(0.001),
                            model::Union{Missing, Flux.Chain} = missing,
                            return_losses::Bool=false,
                            verbose::Bool=true,
                            measures_func::Union{Missing, Function}=missing,
                            return_train_acc::Bool=false,
                            kargs...)
fiddl(X_train::Union{SparseMatrixCSC,Matrix},
              Y_train::Union{SparseMatrixCSC,Matrix},
              learn_seq::Vector,
              data::DataFrame,
              target_col::Union{Symbol, String},
              model_outpath::String;
              hidden_dim::Int=1000,
              batchsize::Int=64,
              loss_func::Function=Flux.mse,
              optimizer=Flux.Adam(0.001),
              model::Union{Missing, Chain} = missing,
              return_losses::Bool=false,
              verbose::Bool=true,
              n_batch_eval::Int=100,
              measures_func::Union{Function, Missing}=missing,
              kargs...)

```
