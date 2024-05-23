from util.dataset import *
from util.model import *
import itertools

def main():
    DIR = "./data/dataRaw.csv"

    # Initialize hyperparameters
    parameters = {
        "LSTM": {
            "input_chunk_length": [1, 31],
            "training_length": [32],
            "hidden_dim": [8, 32],
            "n_rnn_layers": [1, 2, 3],
            "dropout": [0, 0.3],
            "batch_size": [32, 64, 128, 256],
            "lr": [1e-5, 1e-3],
            "epochs" : [100],
            'valEpochs' : [10]
        },
        "GRU": {
            "input_chunk_length": [1, 31],
            "training_length": [62],
            "hidden_dim": [8, 32],
            "n_rnn_layers": [1, 2, 3],
            "dropout": [0, 0.3],
            "batch_size": [32, 64, 128, 256],
            "lr": [1e-5, 1e-3],
            "epochs" : [100],
            'valEpochs' : [10]
        },
        "TFT": {
            "input_chunk_length": [1, 31],
            "output_chunk_length": [1, 31],
            "hidden_size": [8, 32],
            "num_attention_heads" : [2, 8],
            "lstm_layers": [1, 2, 3],
            "dropout": [0, 0.3],
            "hidden_continuous_size": [6, 10],
            "batch_size": [32, 64, 128, 256],
            "full_attention": [True, False],
            "lr": [1e-5, 1e-3],
            "epochs" : [100],
            'valEpochs' : [10]
        }
    }

    # Set model list
    model_list = ["LSTM", "GRU", "TFT"]
    outlier_handling_options = [0, 1]
    cov_options = [False, True]

    for outlierHandling, cov, modelName in itertools.product(outlier_handling_options, cov_options, model_list):
        # Initialize dataset object and load data
        data = DatasetInterface()
        data.initialize_dataset(DIR)

        # Preprocessing
        data.handle_outlier(type = outlierHandling, lenWin = 30) # Handle outliers (0 for IQR, 1 and lenWin > 1 for Isolation Forest)
        data.data_normalization(lenDiff = 1) # Normalize data using Pandas difference method

        # Create timeseries data with a split ratio of 80% training and 20% validation
        data.create_timeseries(split = 0.8)

        # Initialize a model object and set the training settings
        model = ModelInterface(modelName)
        model.trialAmount = 100
        model.useStaticCovariates = cov
        model.verbose = True
        model.errorMetricsName = "rmse" #Choose between rmse, mae, and mape
        model.initializeData(data.target, data.trainTarget, data.valTarget, data.feature)

        # Set the hyperparameters, train the model, and save the data
        print("\n\nstarting searching\n\n")
        model.parameter = parameters[modelName]
        model.train()
        model.saveResult(f"{modelName}-useStaticCovariates={cov}-outlierHandling{outlierHandling}")

        print("\n\nbest model\n\n")
        # Getting the best model and then doing backtest to validate the score
        model.get_best_model(backtest = True, start = 0.8, lenBacktest = 4)

if __name__ == "__main__":
    main()