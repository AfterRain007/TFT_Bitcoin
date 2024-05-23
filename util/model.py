import numpy as np
import pandas as pd
import time
import torch
import optuna
import matplotlib.pyplot as plt
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
        self.fileName = ""
        """file: Name of the file to save the results to"""

        self.target     = None
        """Darts Time Series: Target Dataset"""
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
        self.resultDataframe = pd.DataFrame()
        """Pandas Dataframe: To Store the result of the hyperparameter search"""
        self.trainingTime = []
        """List: To Store the amount of time it takes to train the model"""
        self.predictingTime = []
        """List: List of amount of time to predict using the model"""
        self.study = None
        """Study: To Store the optuna study format"""
        self.pred = None
        """Darts Time Series: Result of prediction using the model"""

    def initializeData(self, trainData, valData, covariateData, target):
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

        # Assign feature data
        self.target = target

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
        self.fileName = fileName
        self.resultDataframe = pd.DataFrame(self.result)
        
        # Add training time, predicting time, and model name columns to the DataFrame
        self.resultDataframe['training_time'] = self.trainingTime
        self.resultDataframe['predicting_time'] = self.predictingTime
        self.resultDataframe['model'] = self.modelName
        self.resultDataframe['epochs'] = self.parameter['epochs'][0]
        self.resultDataframe['valEpochs'] = self.parameter['valEpochs'][0]

        # Retraining and saving the best model
        
        # Save the DataFrame to a CSV file
        self.resultDataframe.to_csv(f"./res/{self.fileName}.csv", index = False)

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
        self.pred = self.model.predict(len(self.ValData))
        self.predictingTime.append(time.time() - start)

        # Calculate RMSE
        rmse_ = rmse(self.ValData, self.pred)

        return rmse_

    def LSTM(self, trial):
        """
        Function to define and train an Long Short-Term Memory (LSTM) model with hyperparameters specified by the Optuna trial.

        :param trial: Optuna trial: The trial object used for hyperparameter optimization
        :return: float: Root Mean Squared Error (RMSE) of the model's predictions on the validation data
        """
        # Initialize Hyperparameters
        input_chunk_length = trial.suggest_int("input_chunk_length", self.parameter['input_chunk_length'][0], self.parameter['input_chunk_length'][1])
        training_length = trial.suggest_int("training_length", input_chunk_length + 1, self.parameter['training_length'][0])
        hidden_dim = trial.suggest_int("hidden_dim", self.parameter['hidden_dim'][0], self.parameter['hidden_dim'][1])
        n_rnn_layers = trial.suggest_categorical("n_rnn_layers", self.parameter['n_rnn_layers'])
        dropout = trial.suggest_float("dropout", self.parameter['dropout'][0], self.parameter['dropout'][1])
        batch_size = trial.suggest_categorical("batch_size", self.parameter['batch_size'])
        lr = trial.suggest_float("lr", self.parameter['lr'][0], self.parameter['lr'][1])

        # Input Hyperparameters into the LSTM Model
        self.model = RNNModel(
            model = "LSTM",
            hidden_dim = hidden_dim,
            n_rnn_layers = n_rnn_layers,
            dropout = dropout,
            batch_size = batch_size,
            training_length = training_length,
            input_chunk_length = input_chunk_length,
            optimizer_kwargs = {"lr": lr},
            n_epochs = self.parameter['epochs'][0],
            nr_epochs_val_period = self.parameter['valEpochs'][0],
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
        input_chunk_length = trial.suggest_int("input_chunk_length", self.parameter['input_chunk_length'][0], self.parameter['input_chunk_length'][1])
        training_length = trial.suggest_int("training_length", input_chunk_length + 1, self.parameter['training_length'][0])
        hidden_dim = trial.suggest_int("hidden_dim", self.parameter['hidden_dim'][0], self.parameter['hidden_dim'][1])
        n_rnn_layers = trial.suggest_categorical("n_rnn_layers", self.parameter['n_rnn_layers'])
        dropout = trial.suggest_float("dropout", self.parameter['dropout'][0], self.parameter['dropout'][1])
        batch_size = trial.suggest_categorical("batch_size", self.parameter['batch_size'])
        lr = trial.suggest_float("lr", self.parameter['lr'][0], self.parameter['lr'][1])

        # Input Hyperparameters into the GRU Model
        self.model = RNNModel(
            model = "GRU",
            hidden_dim = hidden_dim,
            n_rnn_layers = n_rnn_layers,
            dropout = dropout,
            batch_size = batch_size,
            training_length = training_length,
            input_chunk_length = input_chunk_length,
            optimizer_kwargs = {"lr": lr},
            n_epochs = self.parameter['epochs'][0],
            nr_epochs_val_period = self.parameter['valEpochs'][0],
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
        input_chunk_length = trial.suggest_int("input_chunk_length", self.parameter['input_chunk_length'][0], self.parameter['input_chunk_length'][1])
        output_chunk_length = trial.suggest_int("output_chunk_length", self.parameter['output_chunk_length'][0], self.parameter['output_chunk_length'][1])
        hidden_size = trial.suggest_int("hidden_size", self.parameter['hidden_size'][0], self.parameter['hidden_size'][1])
        lstm_layers = trial.suggest_categorical("lstm_layers", self.parameter['lstm_layers'])
        num_attention_heads = trial.suggest_int("num_attention_heads", self.parameter['num_attention_heads'][0], self.parameter['num_attention_heads'][1])
        dropout = trial.suggest_float("dropout", self.parameter['dropout'][0], self.parameter['dropout'][1])
        hidden_continuous_size = trial.suggest_int("hidden_continuous_size", self.parameter['hidden_continuous_size'][0], self.parameter['hidden_continuous_size'][1])
        batch_size = trial.suggest_categorical("batch_size", self.parameter['batch_size'])
        full_attention = trial.suggest_categorical("full_attention", self.parameter['full_attention'])
        lr = trial.suggest_float("lr", self.parameter['lr'][0], self.parameter['lr'][1])

        # Input Hyperparameters into the TFT Model
        self.model = TFTModel(
            input_chunk_length = input_chunk_length,
            output_chunk_length = output_chunk_length,
            hidden_size = hidden_size,
            lstm_layers = lstm_layers,
            num_attention_heads = num_attention_heads,
            dropout = dropout,
            hidden_continuous_size = hidden_continuous_size,
            use_static_covariates = self.useStaticCovariates,
            batch_size = batch_size,
            optimizer_kwargs = {'lr': lr},
            n_epochs = self.parameter['epochs'][0],
            nr_epochs_val_period = self.parameter['valEpochs'][0],
            likelihood = None, 
            loss_fn = torch.nn.MSELoss(),
            full_attention = full_attention,
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

    def get_best_model(self):
        """
        Identifies and saves the best model based on RMSE from the result DataFrame.
        Additionally, it retrains the model, saves the best prediction results to a CSV file,
        and generates a plot comparing predicted prices to true prices.
        """
        
        # Sort the result DataFrame by RMSE in ascending order
        self.resultDataframe.sort_values(by="RMSE", inplace=True)
        
        # Retrieve the parameters of the best model (with the lowest RMSE) as a dictionary
        self.parameter = self.resultDataframe.iloc[0].to_dict()
        
        # Convert each parameter value to a list containing two identical values
        for key, value in self.parameter.items():
            self.parameter[key] = [value, value]
        
        # Set the number of trials to 1 for retraining the best model
        self.trialAmount = 1
        
        # Retrain the model with the best parameters
        self.train()
        
        # Generate the best prediction DataFrame
        bestPrediction = self.pred.pd_dataframe()
        
        # Add the true values to the best prediction DataFrame
        bestPrediction["true"] = self.ValData.pd_dataframe()
        
        # Save the best prediction DataFrame to a CSV file
        bestPrediction.to_csv(f"./best_result/{self.fileName}.csv")
        
        # Plot predicted prices and true prices
        plt.plot(bestPrediction['price'], label='Predicted Price')
        plt.plot(bestPrediction['true'], label='True Price')
        
        # Add title and labels to the plot
        plt.title('Predicted vs True Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        
        # Save the plot to a PNG file
        plt.savefig(f"./plot/{self.fileName}.png")

    def back_test(self):
        backTest = self.model.backtest(series = self.target,
                                       future_covariates = self.covariateData,
                                       start = 0.8,
                                       forecast_horizon = self.,
                                       stride = 102,
                                       verbose = True,
                                       retrain = True,
                                       metric = [rmse, mae, mape],
                                       reduction = np.mean,
                                       last_points_only = False
                                      )

        backTestResult.append(backTest)
        dfBackTest = pd.DataFrame(backTestResult)