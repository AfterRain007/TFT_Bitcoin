import pandas as pd
import numpy as np
import darts
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.ensemble import IsolationForest

class DatasetInterface:
    def __init__(self):
        self.df = None
        self.target = None
        self.trainTarget = None
        self.valTarget = None

        self.feature = None
        self.trainFeature = None
        self.valFeature = None

        self.timeCovariate = None
        self.trainTimeCovariate = None
        self.valTimeCovariate = None

    def initialize_dataset(self, DIR):
        self.df = pd.read_csv(DIR, parse_dates=['date'], index_col = ['date'], usecols = ['date', 'trend', 'price', 'volume', 'sen'])

    def data_normalization(self, lenDiff = 0):
        if lenDiff < 1:
            print("Skipping Data Normalization")
        else:
            self.df.diff(periods = lenDiff)[lenDiff:]

    def handle_outlier(self, type = -1, lenWin = -1):
        columns = ['price', 'volume']
        if type == 0:
            # Using InterQuartile Range (boxplot) Method
            for column in columns:
                q1 = self.df[column].quantile(0.25)
                q3 = self.df[column].quantile(0.75)
                iqr = q3 - q1
            
                # Define the lower and upper bounds for outlier detection
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
            
                # Identify outliers
                outliers_upper = (self.df[column] > upper_bound)
                outliers_under = (self.df[column] < lower_bound)
            
                # Handle outliers by replacing them with the bound value
                self.df.loc[outliers_upper, column] = upper_bound
                self.df.loc[outliers_under, column] = lower_bound

        elif type == 1:
            if lenWin > 1:
                raise TypeError("Outlier Handling with Isolation Forest need argument windows length of > 1")
            # Isolation Forest & Rolling Average Method
            model_IF = IsolationForest(contamination=float(0.1),random_state=42069)
            model_IF.fit(self.df[columns])
            self.df['anomaly_scores'] = model_IF.decision_function(self.df[columns])
            self.df['anomaly'] = model_IF.predict(self.df[columns])

            self.df['repVolume'] = self.df['volume'].rolling(window=lenWin).mean()
            self.df['repPrice'] = self.df['price'].rolling(window=lenWin).mean()
            
            dfAnomaly  = self.df[self.df['anomaly'] == -1]
            dfAnomaly = dfAnomaly[dfAnomaly.index < self.df.iloc[int(len(self.df)*.8)].name].sort_values(by = 'anomaly_scores', ascending = True)
            dfAnomaly = dfAnomaly.iloc[:int(len(dfAnomaly)*.1)]

            self.df.loc[dfAnomaly.index, 'price']  =  self.df['repPrice'].loc[dfAnomaly.index].values
            self.df.loc[dfAnomaly.index, 'volume'] = self.df['repVolume'].loc[dfAnomaly.index].values
            self.df.drop(['anomaly_scores', 'anomaly', 'repPrice', 'repVolume'], axis = 1, inplace = True)
        else:
            print("Skipping Outlier Handling")
            
    def create_timeseries(self, split = 0.8):
        self.target = TimeSeries.from_dataframe(self.df[['price']])
        self.feature = TimeSeries.from_dataframe(self.df[['sen', 'volume', 'trend']])
        
        # train/test split and scaling of target variable
        self.trainTarget, self.valTarget = self.target.split_after(split)
        
        scalerP = Scaler()
        scalerP.fit_transform(self.trainTarget)
        
        self.trainTarget = scalerP.transform(self.trainTarget)
        self.valTarget  = scalerP.transform(self.valTarget)
        # ts_t = scalerP.transform(ts_P)
        
        # make sure data are of type float
        # ts_t = ts_t.astype(np.float32)
        self.trainTarget = self.trainTarget.astype(np.float32)
        self.valTarget = self.valTarget.astype(np.float32)
        
        # train/test split and scaling of feature covariates
        self.trainFeature, self.valFeature = self.feature.split_after(split)
        
        scalerF = Scaler()
        scalerF.fit_transform(self.trainFeature)
        self.trainFeature = scalerF.transform(self.trainFeature)
        self.valFeature  = scalerF.transform(self.valFeature)
        
        self.initialize_time_covariate(split)

        # make sure data are of type float
        self.feature = self.feature.astype(np.float32)
        self.trainFeature = self.trainFeature.astype(np.float32)
        self.valFeature = self.valFeature.astype(np.float32)


    def initialize_time_covariate(self, split):
        # feature engineering - create time covariates: hour, weekday, month, year, country-specific holidays
        covT = datetime_attribute_timeseries(self.feature, attribute="day", one_hot=False, add_length=0)
        covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="week"))
        covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="month"))
        covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="year"))
        covT = covT.stack(TimeSeries.from_times_and_values(times=covT.time_index, values=np.arange(len(self.feature)), columns=["linear_increase"]))
        covT = covT.add_holidays(country_code="US")
        covT = covT.astype(np.float32)
        
        # train/test split
        covT_train, covT_val = covT.split_after(split)
        
        # combine feature and time covariates along component dimension: axis=1
        # ts_cov = ts_covF.concatenate( covT.slice_intersect(self.feature), axis=1 )                      # unscaled F+T

        # rescale the covariates: fitting on the training set   
        scalerT = Scaler()
        scalerT.fit(covT_train)
        self.timeCovariate = scalerT.transform(covT)
        self.trainTimeCovariate = scalerT.transform(covT_train)
        self.valTimeCovariate = scalerT.transform(covT_val)

        self.feature = self.feature.stack(self.timeCovariate)
        self.trainFeature = self.trainFeature.stack(self.trainTimeCovariate)
        self.valFeature = self.valFeature.stack(self.valTimeCovariate)