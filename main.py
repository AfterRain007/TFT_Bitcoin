from util.dataset import *

def main():
    DIR = "./data/dataRaw.csv"

    data = DatasetInterface()
    data.initialize_dataset(DIR)
    data.handle_outlier(type = 0)
    data.data_normalization(lenDiff = 30)
    data.create_timeseries(split = 0.8, useCovariates = True)

    

if __name__ == "__main__":
    main()