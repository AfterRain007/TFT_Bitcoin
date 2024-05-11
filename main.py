from util.dataset import *
from util.model import *

def main():
    DIR = "./data/dataRaw.csv"

    modelList = ["LSTM", "GRU", "TFT"]

    parameters = {
        "LSTM": {
            "inputChunkLength": [1, 31],
            "trainingLength": [62],
            "hiddenDim": [8, 32],
            "nRnnLayers": [1, 2, 3],
            "dropout": [0, 0.3],
            "batchSize": [32, 64, 128, 256],
            "learningRate": [1e-5, 1e-3],
            "epochs" : 2,
            'valEpochs' : 1
        },
        "GRU": {
            "inputChunkLength": [1, 31],
            "trainingLength": [62],
            "hiddenDim": [8, 32],
            "nRnnLayers": [1, 2, 3],
            "dropout": [0, 0.3],
            "batchSize": [32, 64, 128, 256],
            "learningRate": [1e-5, 1e-3],
            "epochs" : 2,
            'valEpochs' : 1
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
            "epochs" : 2,
            'valEpochs' : 1
        },
        "TCN": {
            "inputLength": [2, 31],
            "numFilters": [2, 8],
            'weightNorm' : [True, False],
            "dilationBase": [1, 5],
            "numLayers": [1, 5],
            "dropout": [0, 0.3],
            "batchSize": [32, 64, 128, 256],
            "learningRate": [1e-5, 1e-3],
            "epochs" : 20,
            'valEpochs' : 5
        }
    }
    
    for cov in [False, True]:
        for modelName in modelList:
            data = DatasetInterface()
            data.initialize_dataset(DIR)
            data.handle_outlier(type = 0)
            data.data_normalization(lenDiff = 30)
            data.create_timeseries(split = 0.8)

            model = ModelInterface(modelName)
            model.trialAmount = 1
            model.useStaticCovariates = cov
            model.verbose = False
            model.initializeData(data.trainTarget, data.valTarget, data.feature)
            model.parameter = parameters[modelName]
            model.train()   
            model.saveResult()

if __name__ == "__main__":
    main()