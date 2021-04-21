# TimeSeries : AutoEncoding and Forecasting
Improving existing algorithms with attention
***
## Installation
$ git clone https://github.com/hugoschnoering2/forcasttention.git

## Models
### RNN
#### AutoEncoding
$ python "models/rnn/train_ae.py" -c "models/rnn/config.yaml"

__Results__
Hyperparameters:
* seq_length : 5
* step_size : 24
* embed_size : 10
* learning_rate : 1e-5
* batch_size : 32
* num_epochs : 500

Model | num_layers_encoder | num_layers_decoder | MSE on test set | SoftDTW on test set
--- | --- | ---| --- | ---
GRU all | 1 | 1 | 4.6e-5 | -51


#### Forecasting
***
### TCN
***
### Transformer
