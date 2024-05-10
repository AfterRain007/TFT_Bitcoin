from util.dataset import *
from util.model import *

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError
import time

def main():
    DIR = "./data/dataRaw.csv"

    data = DatasetInterface()
    data.initialize_dataset(DIR)
    data.handle_outlier(type = 0)
    data.data_normalization(lenDiff = 30)
    data.create_timeseries(split = 0.8, useCovariates = True)

    # print(data.covariate)
    model = ModelInterface("LSTM")
    model.trainData = data.train_target
    model.ValData = data.test_target
    model.covariateData = data.covariate

    model.parameter = {
        "inputChunkLength" : [1, 31],
        'trainingLength' : [62],
        'hiddenDim' : [8, 32],
        'nRnnLayers' : [1, 2, 3],
        'dropout' : [0, 0.3],
        'batchSize' : [32, 64, 128, 256],
        'learningRate' : [1e-5, 1e-3]
    }

    model.train()

if __name__ == "__main__":
    main()