from darts.metrics import rmse
from darts import TimeSeries
from darts.models import TFTModel, RNNModel, TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanSquaredError
import numpy as np
import time
import torch
import pandas as pd

class ModelInterface:
    def __init__(self, name):
        self.model = None
        self.modelName = name
        self.parameter = {}

        self.trainData     = None
        self.ValData       = None
        self.covariateData = None

        self.useStaticCovariates = False
        self.trialAmount = 10
        self.verbose = True
        self.result = []

        self.plTrainerKwargs = {
            "accelerator": "auto",
            "callbacks": [EarlyStopping(monitor="val_loss",
                                        patience=10,
                                        min_delta=0.01,
                                        mode='min')]}

    def initializeData(self, trainData, valData, covariateData):
        if self.useStaticCovariates:
            self.covariateData = covariateData
        else:
            self.covariateData = covariateData[['volume', "sen", 'trend']]

        self.trainData     = trainData
        self.ValData       = valData

    def train(self):
        model_functions = {
            'LSTM': self.LSTM,
            'TFT' : self.TFT,
            'GRU' : self.GRU,
        }

        study = optuna.create_study(directions=["minimize"])
        study.optimize(model_functions[self.modelName], n_trials=self.trialAmount, callbacks=[self.print_callback])  

    def print_callback(self, study, trial):
            print(f"Current value: {trial.value}, Current params: {trial.params}")
            print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

            # Saving results into a list
            temp = trial.params
            temp.update({"RMSE" : trial.value})
            self.result.append(temp)

    def saveResult(self, fileName):
        result = pd.DataFrame(self.result)
        result.to_csv(fileName)

    def trainAndTest(self):
        # Training the model
        self.model.fit(series = self.trainData,                     # Train Price
                       val_series = self.ValData,                   # Val Price
                       future_covariates = self.covariateData,      # Val Covariate
                       val_future_covariates = self.covariateData,  # Val Covariate
                       verbose = self.verbose)

        # Predict using the model
        pred = self.model.predict(len(self.ValData))

        # Calculate RMSE
        rmse_ = rmse(self.ValData, pred)

        return rmse_

    def LSTM(self,trial):
        # Initialize Hyperparameters
        inputChunkLength = trial.suggest_int("input_chunk_length", self.parameter['inputChunkLength'][0], self.parameter['inputChunkLength'][1])
        trainingLength = trial.suggest_int("training_length", inputChunkLength + 1, self.parameter['trainingLength'][0])
        hiddenDim = trial.suggest_int("hidden_dim", self.parameter['hiddenDim'][0], self.parameter['hiddenDim'][1])
        nRnnLayers = trial.suggest_categorical("n_rnn_layers", self.parameter['nRnnLayers'])
        dropout = trial.suggest_float("dropout", self.parameter['dropout'][0], self.parameter['dropout'][1])
        batchSize = trial.suggest_categorical("batch_size", self.parameter['batchSize'])
        learningRate = trial.suggest_float("lr", self.parameter['learningRate'][0], self.parameter['learningRate'][1])

        # Input Hyperparameters into the LSTM Model
        self.model = RNNModel(
            model = "LSTM",
            hidden_dim = hiddenDim,
            n_rnn_layers = nRnnLayers,
            dropout = dropout,
            batch_size = batchSize,
            training_length = trainingLength,
            input_chunk_length = inputChunkLength,
            optimizer_kwargs = {"lr": learningRate},
            n_epochs = self.parameter['epochs'],
            nr_epochs_val_period = self.parameter['valEpochs'],
            torch_metrics = MeanSquaredError(squared = False),
            log_tensorboard = True,
            random_state = 42069,
            force_reset = True,
            save_checkpoints = False,
            pl_trainer_kwargs = self.plTrainerKwargs
        )

        rmse_ = self.trainAndTest()

        return rmse_ if rmse_ != np.nan else float("inf")

    def GRU(self,trial):
        # Initialize Hyperparameters
        inputChunkLength = trial.suggest_int("input_chunk_length", self.parameter['inputChunkLength'][0], self.parameter['inputChunkLength'][1])
        trainingLength = trial.suggest_int("training_length", inputChunkLength + 1, self.parameter['trainingLength'][0])
        hiddenDim = trial.suggest_int("hidden_dim", self.parameter['hiddenDim'][0], self.parameter['hiddenDim'][1])
        nRnnLayers = trial.suggest_categorical("n_rnn_layers", self.parameter['nRnnLayers'])
        dropout = trial.suggest_float("dropout", self.parameter['dropout'][0], self.parameter['dropout'][1])
        batchSize = trial.suggest_categorical("batch_size", self.parameter['batchSize'])
        learningRate = trial.suggest_float("lr", self.parameter['learningRate'][0], self.parameter['learningRate'][1])

        # Input Hyperparameters into the GRU Model
        self.model = RNNModel(
            model = "GRU",
            hidden_dim = hiddenDim,
            n_rnn_layers = nRnnLayers,
            dropout = dropout,
            batch_size = batchSize,
            training_length = trainingLength,
            input_chunk_length = inputChunkLength,
            optimizer_kwargs = {"lr": learningRate},
            n_epochs = self.parameter['epochs'],
            nr_epochs_val_period = self.parameter['valEpochs'],
            torch_metrics = MeanSquaredError(squared = False),
            log_tensorboard = True,
            random_state = 42069,
            force_reset = True,
            save_checkpoints = False,
            pl_trainer_kwargs = self.plTrainerKwargs
        )

        rmse_ = self.trainAndTest()

        return rmse_ if rmse_ != np.nan else float("inf")

    # define objective function
    def TFT(self, trial):
        # Initialize Hyperparameters
        inputLength = trial.suggest_int("in_len", self.parameter['inputLength'][0], self.parameter['inputLength'][1])
        outputLength = trial.suggest_int("out_len", self.parameter['outputLength'][0], self.parameter['outputLength'][1])
        hiddenSize = trial.suggest_int("hidden_size", self.parameter['hiddenSize'][0], self.parameter['hiddenSize'][1])
        LSTMLayers = trial.suggest_categorical("lstm_layers", self.parameter['LSTMLayers'])
        NumAttentionHeads = trial.suggest_int("num_attention_heads", self.parameter['NumAttentionHeads'][0], self.parameter['NumAttentionHeads'][1])
        dropout = trial.suggest_float("dropout", self.parameter['dropout'][0], self.parameter['dropout'][1])
        hiddenContinuousSize = trial.suggest_int("hidden_continuous_size", self.parameter['hiddenContinuousSize'][0], self.parameter['hiddenContinuousSize'][1])
        batchSize = trial.suggest_categorical("batch_size", self.parameter['batchSize'])
        fullAttention = trial.suggest_categorical("full_attention", self.parameter['fullAttention'])
        learningRate = trial.suggest_float("lr", self.parameter['learningRate'][0], self.parameter['learningRate'][1])

        # Input Hyperparameters into the TFT Model
        self.model = TFTModel(
            input_chunk_length = inputLength,
            output_chunk_length = outputLength,
            hidden_size = hiddenSize,
            lstm_layers = LSTMLayers,
            num_attention_heads = NumAttentionHeads,
            dropout = dropout,
            hidden_continuous_size = hiddenContinuousSize,
            use_static_covariates = self.useStaticCovariates,
            batch_size = batchSize,
            optimizer_kwargs = {'lr': learningRate},
            n_epochs = self.parameter['epochs'],
            nr_epochs_val_period = self.parameter['valEpochs'],
            likelihood = None, 
            loss_fn = torch.nn.MSELoss(),
            full_attention = fullAttention,
            torch_metrics = MeanSquaredError(squared = False),
            random_state = 42069,
            force_reset= True,
            pl_trainer_kwargs = self.plTrainerKwargs,
            add_relative_index= False,
            # add_encoders = None, #Uncomment these two to use Probabilistic Forecast
            # likelihood=QuantileRegression(quantiles=[.01, .05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .75, .8, .85, .90, .95, .99])
        )

        rmse_ = self.trainAndTest()

        return rmse_ if rmse_ != np.nan else float("inf")

    ## There's an error here (The RMSE value is 700000 > and I don't know why soooooooooo)
    # def TCN(self, trial):
    #     # Initialize Hyperparameters
    #     inputLength = trial.suggest_int("in_len", self.parameter["inputLength"][0], self.parameter["inputLength"][1])
    #     outputLength = trial.suggest_int("out_len", 1, inputLength - 1)
    #     kernelSize = trial.suggest_int("kernel_size", 1, inputLength - 1)
    #     numFilters = trial.suggest_int("num_filters", self.parameter['numFilters'][0], self.parameter['numFilters'][1])
    #     weightNorm = trial.suggest_categorical("weight_norm", self.parameter['weightNorm'])
    #     dilationBase = trial.suggest_int("dilation_base", self.parameter['dilationBase'][0], self.parameter['dilationBase'][1])
    #     numLayers = trial.suggest_int("num_layers", self.parameter['numLayers'][0], self.parameter['numLayers'][1])
    #     dropout = trial.suggest_float("dropout", self.parameter['dropout'][0], self.parameter['dropout'][1])
    #     learningRate = trial.suggest_float("lr", self.parameter['learningRate'][0], self.parameter['learningRate'][1])
    #     batchSize = trial.suggest_categorical("batch_size", self.parameter['batchSize'])
        
    #     # Input Hyperparameters into the TCN Model
    #     model = TCNModel(
    #         input_chunk_length = inputLength,
    #         output_chunk_length = outputLength,
    #         kernel_size = kernelSize,
    #         num_filters = numFilters,
    #         weight_norm = weightNorm,
    #         dilation_base = dilationBase,
    #         num_layers = numLayers,
    #         dropout = dropout,
    #         batch_size = batchSize,
    #         optimizer_kwargs = {'lr': learningRate},
    #         n_epochs = self.parameter['epochs'],
    #         nr_epochs_val_period = self.parameter['valEpochs'],
    #         likelihood = None, 
    #         loss_fn = torch.nn.MSELoss(),
    #         torch_metrics = MeanSquaredError(squared = False),
    #         random_state = 42069,
    #         force_reset= True,
    #         pl_trainer_kwargs = self.plTrainerKwargs,
    #     )

    #     # Training the Model
    #     model.fit(series = self.trainData,                   # Train Price
    #               val_series = self.ValData,                 # Val Price
    #               past_covariates = self.covariateData,      # Val Covariate
    #               val_past_covariates = self.covariateData,  # Val Covariate
    #               verbose = self.verbose)

    #     # Predict using the model
    #     pred = model.predict(len(self.ValData))

    #     # Calculate RMSE
    #     rmse_ = rmse(self.ValData, pred)

    #     return rmse_ if rmse_ != np.nan else float("inf")