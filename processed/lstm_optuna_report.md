# Day 11 - Optuna LSTM Optimization

|   sequence_length |   batch_size |   hidden_size |   num_layers |   dropout |   learning_rate |   max_epochs |
|------------------:|-------------:|--------------:|-------------:|----------:|----------------:|-------------:|
|                43 |           16 |            68 |            1 | 0.0200118 |      0.00483627 |           23 |

| metric   |           value |
|:---------|----------------:|
| MAPE     |     1.42623e+13 |
| RMSE     | 24231.3         |
| MAE      | 15975.6         |

The optimized LSTM is chosen on the lowest validation loss before evaluation on the held-out test window.