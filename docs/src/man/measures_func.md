# The `measures_func` argument

The deep learning functions `get_and_train_model` and `fiddl` take a `measures_func` as one of their arguments. This helps computing measures during the training. For this to work, the `measures_func` has to conform to the following format.

## For `get_and_train_model`

```
data_train, data_val = measures_func(X_train,
                                    Y_train,
                                    X_val,
                                    Y_val,
                                    Yhat_train,
                                    Yhat_val,
                                    data_train,
                                    data_val,
                                    target_col,
                                    model,
                                    epoch;
                                    kargs...)

## Input

- `X_train`: The input training matrix.
- `Y_train`: The target training matrix
- `X_val`: The input validation matrix.
- `Y_val`: The target validation matrix.
- `Yhat_train`: The predicted training matrix.
- `Yhat_val`: The predicted validation matrix.
- `data_train`: The training dataset.
- `data_val`: The validation dataset.
- `target_col`: The name of the column with the target wordforms in the datasets.
- `model`: The trained model.
- `epoch`: The epoch the training is currently in.
- `kargs...`: Any other keyword arguments that should be passed to the function.

Note: the `kargs` are just keyword arguments that are passed on from the parameters of `get_and_train_model` to the `measures_func`. For example, this could be a suffix that should be added to each added column in `measures_func`.

## Output
The function has to return the training and validation dataframes.
```

### Example

Define a `measures_func`. This one computes target correlations for both training and validation datasets.
```
function compute_target_corr(X_train, Y_train, X_val, Y_val,
                                    Yhat_train, Yhat_val, data_train,
                                    data_val, target_col, model, epoch)
        _, corr = JudiLing.eval_SC(Yhat_train, Y_train, R=true)
        data_train[!, string("target_corr_", epoch)] = diag(corr)
        _, corr = JudiLing.eval_SC(Yhat_val, Y_val, R=true)
        data_val[!, string("target_corr_", epoch)] = diag(corr)
        return(data_train, data_val)
end
```

Train a model for 100 epochs, call `compute_target_corr` after each epoch.
```
res = JudiLing.get_and_train_model(cue_obj_train.C,
                            S_train,
                            cue_obj_val.C,
                            S_val,
                            train, val,
                            :Word,
                            "test.bson",
                            return_losses=true,
                            batchsize=3,
                            measures_func=compute_target_corr)

```

## For `fiddl`

```
data = measures_func(X_train,
                      Y_train,
                      Yhat_train,
                      data,
                      target_col,
                      model,
                      step;
                      kargs...)

## Input

- `X_train`: The input matrix of the full dataset.
- `Y_train`: The target matrix of the full dataset.
- `Yhat_train`: The predicted matrix of the full dataset at current step.
- `data_train`: The full dataset.
- `target_col`: The name of the column with the target wordforms in the dataset.
- `model`: The trained model.
- `step`: The step the training is currently in.
- `kargs...`: Any other keyword arguments that should be passed to the function.

Note: the `kargs` are just keyword arguments that are passed on from the parameters of `get_and_train_model` to the `measures_func`. For example, this could be a suffix that should be added to each added column in `measures_func`.

## Output
The function has to return the dataset.
```
