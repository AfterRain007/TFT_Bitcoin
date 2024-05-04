def createTimeseries(df):
    ts_P = TimeSeries.from_dataframe(df[['price']])
    ts_covF = TimeSeries.from_dataframe(df[['sen', 'volume', 'trend']])
    
    SPLIT = .8
    # train/test split and scaling of target variable
    ts_train, ts_test = ts_P.split_after(SPLIT)
    
    scalerP = Scaler()
    scalerP.fit_transform(ts_train)
    ts_ttrain = scalerP.transform(ts_train)
    ts_ttest = scalerP.transform(ts_test)
    ts_t = scalerP.transform(ts_P)
    
    # make sure data are of type float
    ts_t = ts_t.astype(np.float32)
    ts_ttrain = ts_ttrain.astype(np.float32)
    ts_ttest = ts_ttest.astype(np.float32)
    
    # train/test split and scaling of feature covariates
    covF_train, covF_test = ts_covF.split_after(SPLIT)
    
    scalerF = Scaler()
    scalerF.fit_transform(covF_train)
    covF_ttrain = scalerF.transform(covF_train)
    covF_ttest = scalerF.transform(covF_test)
    covF_t = scalerF.transform(ts_covF)
    
    # make sure data are of type float
    covF_ttrain = covF_ttrain.astype(np.float32)
    covF_ttest = covF_ttest.astype(np.float32)
    
    return ts_t, ts_ttrain, ts_ttest, cov_t















#Do Differencing Transformation to All Data
def differT(df):

    df.diff()[1:]

    return df

def testingModel(model):
    global predict
    pred = model.predict(len(ts_ttest))
    pred = scalerP.inverse_transform(pred)
    done = inverse_differenced_dataset(dfPure.pd_dataframe(), pred.pd_dataframe()['price'].tolist())
    predict = pd.DataFrame(done)[1:]

    predict.index = test[:len(test)-1].time_index
    predict = TimeSeries.from_dataframe(predict)
    smape_ = smape(predict, test, n_jobs=-1, verbose=True)
    mape_ = mape(predict, test, n_jobs=-1, verbose=True)
    return smape_, mape_

def anomaly_search(df):
    anomaly_inputs = ['price', 'volume']
    model_IF = IsolationForest(contamination=float(0.1),random_state=42069)
    model_IF.fit(df[anomaly_inputs])
    df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
    df['anomaly'] = model_IF.predict(df[anomaly_inputs])
    return df

def replaceOutlierMA(df, i):
    lenWin = i
    df = anomaly_search(df)
    df['anomaly'] = df['anomaly'].map({1:0, -1:1})
    dfAnomaly  = df[df['anomaly'] == 1]

    df['repVolume'] = df['volume'].rolling(window=lenWin).mean()
    df['repPrice'] = df['price'].rolling(window=lenWin).mean()

    dfAnomaly = dfAnomaly[dfAnomaly.index < df.iloc[int(len(df)*.8)].name].sort_values(by = 'anomaly_scores', ascending = True)
    dfAnomaly = dfAnomaly.iloc[:int(len(dfAnomaly)*.1)]

    df2 = df.copy()
    df2.loc[dfAnomaly.index, 'price']  =  df2['repPrice'].loc[dfAnomaly.index].values
    df2.loc[dfAnomaly.index, 'volume'] = df2['repVolume'].loc[dfAnomaly.index].values
    df2.drop(['anomaly_scores', 'anomaly', 'repPrice', 'repVolume'], axis = 1, inplace = True)

    df2 = differT(df2)
    return df2

def replaceOutlierBP(df):
    # Calculate the IQR (Interquartile Range)
    columns = ['price', 'volume']
    for column in columns:
        
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
    
        # Define the lower and upper bounds for outlier detection
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    
        # Identify outliers
        outliers_upper = (df[column] > upper_bound)
        outliers_under = (df[column] < lower_bound)
    
        # Handle outliers by replacing them with the bound value
        df.loc[outliers_upper, column] = upper_bound
        df.loc[outliers_under, column] = lower_bound

    df = differT(df)
    return df