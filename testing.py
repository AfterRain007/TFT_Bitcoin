import pandas as pd
import matplotlib.pyplot as plt

test = pd.read_csv("./best_result/LSTM-useStaticCovariates=False-outlierHandling1.csv", index_col = ["date"], parse_dates=['date'])

plt.plot(test['price'])
# plt.plot(bestPrediction['TRUE'])
plt.title('DJIA Open and Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()