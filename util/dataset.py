import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.ensemble import IsolationForest

class DatasetInterface:
    def __init__(self):
        self.df = None
        self.target = None
        self.feature = None

        self.train_target = None
        self.test_target = None
        self.cov_target = None

        self.train_feature = None
        self.test_feature = None
        self.cov_feature = None

        self.covariate = None

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
                raise TypeError("Handle Outlier for Isolation Forest need argument windows length of > 1")
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
            
    def create_timeseries(self, split = 0.8, useCovariates = False):
        self.target = TimeSeries.from_dataframe(self.df[['price']])
        self.feature = TimeSeries.from_dataframe(self.df[['sen', 'volume', 'trend']])
        
        # train/test split and scaling of target variable
        self.train_target, self.test_target = self.target.split_after(split)
        
        scalerP = Scaler()
        scalerP.fit_transform(self.train_target)
        
        self.train_target = scalerP.transform(self.train_target)
        self.test_target  = scalerP.transform(self.test_target)
        # ts_t = scalerP.transform(ts_P)
        
        # make sure data are of type float
        # ts_t = ts_t.astype(np.float32)
        self.train_target = self.train_target.astype(np.float32)
        self.test_target  = self.test_target.astype(np.float32)
        
        # train/test split and scaling of feature covariates
        self.train_feature, self.test_feature = self.feature.split_after(split)
        
        scalerF = Scaler()
        scalerF.fit_transform(self.train_feature)
        self.train_feature = scalerF.transform(self.train_feature)
        self.test_feature  = scalerF.transform(self.test_feature)
        # covF_t = scalerF.transform(ts_covF)
        
        # make sure data are of type float
        self.train_feature = self.train_feature.astype(np.float32)
        self.test_feature = self.test_feature.astype(np.float32)

        if useCovariates:
            self.initialize_covariate(split)

    def initialize_covariate(self, split):
        # feature engineering - create time covariates: hour, weekday, month, year, country-specific holidays
        covT = datetime_attribute_timeseries(self.target.time_index, attribute="day", one_hot=False, add_length=len(self.train_target))
        covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="week"))
        covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="month"))
        covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="year"))
        covT = covT.stack(TimeSeries.from_times_and_values(times=covT.time_index, values=np.arange(len(self.target) + len(self.train_target)), columns=["linear_increase"]))
        covT = covT.add_holidays(country_code="US")
        covT = covT.astype(np.float32)
        
        # train/test split
        covT_train, covT_test = covT.split_after(split)
        
        # combine feature and time covariates along component dimension: axis=1
        # ts_cov = ts_covF.concatenate( covT.slice_intersect(self.feature), axis=1 )                      # unscaled F+T

        # rescale the covariates: fitting on the training set   
        scalerT = Scaler()
        scalerT.fit(covT_train)
        # covT_ttrain = scalerT.transform(covT_train)
        # covT_ttest = scalerT.transform(covT_test)
        self.covariate = scalerT.transform(covT)
        
        # covT_ttrain = covT_ttrain.astype(np.float32)
        # covT_ttest  = covT_ttest.astype(np.float32)
        self.covariate      = self.covariate.astype(np.float32)