from util.dataset import *
from util.model import *

def main():
    DIR = "./data/dataRaw.csv"

    # Initialize hyperparameters
    parameters = {
        "LSTM": {
            "inputChunkLength": [1, 31],
            "trainingLength": [32],
            "hiddenDim": [8, 32],
            "nRnnLayers": [1, 2, 3],
            "dropout": [0, 0.3],
            "batchSize": [32, 64, 128, 256],
            "learningRate": [1e-5, 1e-3],
            "epochs" : 100,
            'valEpochs' : 10
        },
        "GRU": {
            "inputChunkLength": [1, 31],
            "trainingLength": [62],
            "hiddenDim": [8, 32],
            "nRnnLayers": [1, 2, 3],
            "dropout": [0, 0.3],
            "batchSize": [32, 64, 128, 256],
            "learningRate": [1e-5, 1e-3],
            "epochs" : 100,
            'valEpochs' : 10
        },
        "TFT": {
            "inputLength": [1, 31],
            "outputLength": [1, 31],
            "hiddenSize": [8, 32],
            "NumAttentionHeads" : [2, 8],
            "LSTMLayers": [1, 2, 3],
            "dropout": [0, 0.3],
            "hiddenContinuousSize": [6, 10],
            "batchSize": [32, 64, 128, 256],
            "fullAttention": [True, False],
            "learningRate": [1e-5, 1e-3],
            "epochs" : 100,
            'valEpochs' : 10
        }
    }

    # Set model list
    modelList = ["LSTM", "GRU", "TFT"]

    for cov in [False, True]:
        for modelName in modelList:
            # Initialize dataset object and load data
            data = DatasetInterface()
            data.initialize_dataset(DIR)

            # Preprocessing
            data.handle_outlier(type = 0, lenWin = 0) # Handle outliers (0 for IQR, 1 and lenWin > 1 for Isolation Forest)
            data.data_normalization(lenDiff = 30) # Normalize data using Pandas difference method

            # Create timeseries data with a split ratio of 80% training and 20% validation
            data.create_timeseries(split=0.8)

            # Initialize a model object and set the training settings
            model = ModelInterface(modelName)
            model.trialAmount = 100
            model.useStaticCovariates = cov
            model.verbose = True
            model.initializeData(data.trainTarget, data.valTarget, data.feature)

            # Set the hyperparameters, train the model, and save the data
            model.parameter = parameters[modelName]
            model.train()
            model.saveResult(f"./res/{modelName}-useStaticCovariates={cov}.csv")

if __name__ == "__main__":
    main()