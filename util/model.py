from darts.metrics import mape, mae, rmse
from darts import TimeSeries, concatenate
from darts.models import TFTModel, RNNModel, TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.statistics import check_seasonality, plot_acf, extract_trend_and_seasonality

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError
import time

class ModelInterface:
    def __init__(self, name):
        self.model = None
        self.modelName = name
        self.parameter = {}

        self.trainData     = None
        self.ValData       = None
        self.covariateData = None

        self.time = None

        self.verbose = True

    def initializeData(self):
        self.trainData = data_train

    def objectiveLSTM(self,trial):
        my_stopper = EarlyStopping(monitor="val_loss",
                                    patience=10,
                                    min_delta=0.01,
                                    mode='min',)
        pl_trainer_kwargs = {
            "accelerator": "auto",
            "callbacks": [my_stopper]}
        
        # select input and output chunk lengths
        inputChunkLength = trial.suggest_int("input_chunk_length", self.parameter['inputChunkLength'][0], self.parameter['inputChunkLength'][1])
        trainingLength = trial.suggest_int("training_length", inputChunkLength + 1, self.parameter['trainingLength'][0])

        # Other hyperparameters
        hiddenDim = trial.suggest_int("hidden_dim", self.parameter['hiddenDim'][0], self.parameter['hiddenDim'][1])
        nRnnLayers = trial.suggest_categorical("n_rnn_layers", self.parameter['nRnnLayers'])
        dropout = trial.suggest_float("dropout", self.parameter['dropout'][0], self.parameter['dropout'][1])
        batchSize = trial.suggest_categorical("batch_size", self.parameter['batchSize'])
        learningRate = trial.suggest_float("lr", self.parameter['learningRate'][0], self.parameter['learningRate'][1])

        # build the LSTM model
        model = RNNModel(
            model = "LSTM",
            hidden_dim = hiddenDim,
            n_rnn_layers = nRnnLayers,
            dropout = dropout,
            batch_size = batchSize,
            training_length = trainingLength,
            input_chunk_length = inputChunkLength,
            optimizer_kwargs = {"lr": learningRate},
            n_epochs = 500,
            nr_epochs_val_period = 10,
            torch_metrics = MeanAbsolutePercentageError(),
            log_tensorboard = True,
            random_state = 42069,
            force_reset = True,
            save_checkpoints = False,
            pl_trainer_kwargs = pl_trainer_kwargs
        )

        # when validating during training, we can use a slightly longer validation
        # set which also contains the first input_chunk_length time steps
        # model_val_set = scaler.transform(series[-(VAL_LEN + in_len) :])

        # train the model
        model.fit(self.trainData,                            # Train Price
                  val_series=self.ValData,                   # Val Price
                  future_covariates=self.covariateData,      # Val Covariate
                  val_future_covariates=self.covariateData,  # Val Covariate
                  verbose=self.verbose)

        return rmse_ if rmse_ != np.nan else float("inf")

    def train(self):
        def print_callback(study, trial):
            print(f"Current value: {trial.value}, Current params: {trial.params}")
            print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

            temp = trial.params
            temp.update({"RMSE" : trial.value})

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objectiveLSTM, n_trials=100, callbacks=[print_callback])