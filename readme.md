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
* learning_rate : 1e-5
* batch_size : 32
* num_epochs : 500

Model | num_layers_encoder | num_layers_decoder | Embed size | MSE on test set | SoftDTW on test set
--- | --- | ---| --- | ---



#### Forecasting
***
### TCN
$ python "models/tcn/train_ae.py" -c "models/tcn/config.yaml"
***
### Transformer
