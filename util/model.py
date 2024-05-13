import numpy as np
import pandas as pd
import time
import torch
import optuna
from darts.metrics import rmse
from darts.models import TFTModel, RNNModel
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanSquaredError

class ModelInterface:
    """
    Interface for managing machine learning models.
    """
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        self.model = None
        """Model: Variable to Train the model at"""
        self.modelName = name
        """String: Model name that's used"""
        self.parameter = {}
        """Dictionary: Dictionary of hyperparameters search space"""

        self.trainData     = None
        """Darts Time Series: Target Training Dataset"""
        self.ValData       = None
        """Darts Time Series: Target Validation Dataset"""
        self.covariateData = None
        """Darts Time Series: Covariate Dataset"""

        self.useStaticCovariates = False
        """Bool: Whether to use Static Covariates or not"""
        self.trialAmount = 10
        """Int: Amount of trials for Hyperparameter Tuning"""
        self.verbose = True
        """Bool: Whether to Print output of the training phase or not"""
        self.result = []
        """List: To Store the result of the hyperparameter search"""
        self.trainingTime = []
        """List: To Store the amount of time it takes to train the model"""
        self.predictingTime = []
        """List: List of amount of time to predict using the model"""
        self.study = None
        """Study: To Store the amount of time to predict using the model"""

    def initializeData(self, trainData, valData, covariateData):
        """
        Initializes the data for training/validation and covariates if applicable.
        
        :param trainData: DataFrame: Training data
        :param valData: DataFrame: Validation data
        :param covariateData: DataFrame: Covariate data
        """
        # If static covariates are used, assign all covariate data
        if self.useStaticCovariates:
            self.covariateData = covariateData
        # If not using static covariates, select specific covariates
        else:
            self.covariateData = covariateData[['volume', "sen", 'trend']]

        # Assign training data
        self.trainData = trainData
        # Assign validation data
        self.ValData = valData

    def train(self):
        """
        Trains the model using Optuna study with specified model functions.
        """
        # Dictionary mapping model names to their corresponding functions
        model_functions = {
            'LSTM': self.LSTM,
            'TFT': self.TFT,
            'GRU': self.GRU,
        }

        # Create an Optuna study for hyperparameter optimization
        self.study = optuna.create_study(directions=["minimize"])
        
        # Optimize the specified model function using Optuna
        self.study.optimize(
            model_functions[self.modelName],   # Function to optimize for the current model
            n_trials=self.trialAmount,         # Number of trials for optimization
            callbacks=[self.print_callback]    # Optional callback function for logging
        )
 

    def print_callback(self, study, trial):
        """
        Callback function to print current and best values and parameters during Optuna study.
        
        :param study: Optuna study: The study object being optimized
        :param trial: Optuna trial: The current trial object
        """
        # Print current value and parameters of the trial
        print(f"Current value: {trial.value}, Current params: {trial.params}")
        # Print best value and parameters found so far in the study
        print(f"Best value: {self.study.best_value}, Best params: {self.study.best_trial.params}")

        # Saving results into a list
        temp = trial.params
        temp.update({"RMSE": trial.value})
        self.result.append(temp)

    def saveResult(self, fileName):
        """
        Save the results to a CSV file, including model parameters and performance metrics.
        
        :param fileName: str: Name of the file to save the results to
        """
        # Convert the list of results into a DataFrame
        result = pd.DataFrame(self.result)
        
        # Add training time and predicting time columns to the DataFrame
        result['training_time'] = self.trainingTime
        result['predicting_time'] = self.predictingTime
        
        # Save the DataFrame to a CSV file
        result.to_csv(fileName)

    def trainAndTest(self):
        """
        Train the model on the training data and evaluate its performance on the validation data.
        Timing for training and predicting is also recorded.
        
        :return: float: Root Mean Squared Error (RMSE) of the model's predictions on the validation data
        """
        # Training the model and timing it
        start = time.time()
        self.model.fit(series=self.trainData,                     # Train Price
                    val_series=self.ValData,                   # Validation Price
                    future_covariates=self.covariateData,      # Validation Covariate
                    val_future_covariates=self.covariateData,  # Validation Covariate
                    verbose=self.verbose)
        self.trainingTime.append(time.time() - start)

        # Predict using the model and timing it
        start = time.time()
        pred = self.model.predict(len(self.ValData))
        self.predictingTime.append(time.time() - start)

        # Calculate RMSE
        rmse_ = rmse(self.ValData, pred)

        return rmse_

    def LSTM(self, trial):
        """
        Function to define and train an Long Short-Term Memory (LSTM) model with hyperparameters specified by the Optuna trial.

        :param trial: Optuna trial: The trial object used for hyperparameter optimization
        :return: float: Root Mean Squared Error (RMSE) of the model's predictions on the validation data
        """
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
            pl_trainer_kwargs = {"accelerator": "auto",
                                 "callbacks": [EarlyStopping(monitor="val_loss",
                                                             patience=10,
                                                             min_delta=0.01,
                                                             mode='min')]}
            )
        
        rmse_ = self.trainAndTest()

        return rmse_ if rmse_ != np.nan else float("inf")

    def GRU(self, trial):
        """
        Function to define and train a Gated Recurrent Units (GRU) model with hyperparameters specified by the Optuna trial.

        :param trial: Optuna trial: The trial object used for hyperparameter optimization
        :return: float: Root Mean Squared Error (RMSE) of the model's predictions on the validation data
        """
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
            pl_trainer_kwargs = {"accelerator": "auto",
                                 "callbacks": [EarlyStopping(monitor="val_loss",
                                                             patience=10,
                                                             min_delta=0.01,
                                                             mode='min')]}
            )

        rmse_ = self.trainAndTest()

        return rmse_ if rmse_ != np.nan else float("inf")

    def TFT(self, trial):
        """
        Function to define and train a Temporal Fusion Transformer (TFT) model with hyperparameters specified by the Optuna trial.

        :param trial: Optuna trial: The trial object used for hyperparameter optimization
        :return: float: Root Mean Squared Error (RMSE) of the model's predictions on the validation data
        """
        # Initialize Hyperparameters
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
            pl_trainer_kwargs = {"accelerator": "auto",
                                 "callbacks": [EarlyStopping(monitor="val_loss",
                                                             patience=10,
                                                             min_delta=0.01,
                                                             mode='min')]},
            add_relative_index= False,
            # add_encoders = None, #Uncomment these two to use Probabilistic Forecast
            # likelihood=QuantileRegression(quantiles=[.01, .05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .75, .8, .85, .90, .95, .99])
        )

        rmse_ = self.trainAndTest()

        return rmse_ if rmse_ != np.nan else float("inf")

    ## There's an error here (The RMSE value is 700000 > and I don't know why soooooooooo)
    # def TCN(self, trial):
        """
        Function to define and train a Temporal Fusion Transformer (TFT) model with hyperparameters specified by the Optuna trial.

        :param trial: Optuna trial: The trial object used for hyperparameter optimization
        :return: float: Root Mean Squared Error (RMSE) of the model's predictions on the validation data
        """
    #   # Initialize Hyperparameters
    #   inputLength = trial.suggest_int("in_len", self.parameter["inputLength"][0], self.parameter["inputLength"][1])
    #   outputLength = trial.suggest_int("out_len", 1, inputLength - 1)
    #   kernelSize = trial.suggest_int("kernel_size", 1, inputLength - 1)
    #   numFilters = trial.suggest_int("num_filters", self.parameter['numFilters'][0], self.parameter['numFilters'][1])
    #   weightNorm = trial.suggest_categorical("weight_norm", self.parameter['weightNorm'])
    #   dilationBase = trial.suggest_int("dilation_base", self.parameter['dilationBase'][0], self.parameter['dilationBase'][1])
    #   numLayers = trial.suggest_int("num_layers", self.parameter['numLayers'][0], self.parameter['numLayers'][1])
    #   dropout = trial.suggest_float("dropout", self.parameter['dropout'][0], self.parameter['dropout'][1])
    #   learningRate = trial.suggest_float("lr", self.parameter['learningRate'][0], self.parameter['learningRate'][1])
    #   batchSize = trial.suggest_categorical("batch_size", self.parameter['batchSize'])
      
    #   # Input Hyperparameters into the TCN Model
    #   model = TCNModel(
    #       input_chunk_length = inputLength,
    #       output_chunk_length = outputLength,
    #       kernel_size = kernelSize,
    #       num_filters = numFilters,
    #       weight_norm = weightNorm,
    #       dilation_base = dilationBase,
    #       num_layers = numLayers,
    #       dropout = dropout,
    #       batch_size = batchSize,
    #       optimizer_kwargs = {'lr': learningRate},
    #       n_epochs = self.parameter['epochs'],
    #       nr_epochs_val_period = self.parameter['valEpochs'],
    #       likelihood = None, 
    #       loss_fn = torch.nn.MSELoss(),
    #       torch_metrics = MeanSquaredError(squared = False),
    #       random_state = 42069,
    #       force_reset= True,
    #       pl_trainer_kwargs = {"accelerator": "auto",
    #                            "callbacks": [EarlyStopping(monitor="val_loss",
    #                                                        patience=10,
    #                                                        min_delta=0.01,
    #                                                        mode='min')]},
    #   )

    #   # Training the Model
    #   model.fit(series = self.trainData,                   # Train Price
    #             val_series = self.ValData,                 # Val Price
    #             past_covariates = self.covariateData,      # Val Covariate
    #             val_past_covariates = self.covariateData,  # Val Covariate
    #             verbose = self.verbose)

    #   # Predict using the model
    #   pred = model.predict(len(self.ValData))

    #   # Calculate RMSE
    #   rmse_ = rmse(self.ValData, pred)

    #   return rmse_ if rmse_ != np.nan else float("inf")