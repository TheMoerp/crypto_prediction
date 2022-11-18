import requests
import pandas as pd
import json
import matplotlib as plt
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
from pycaret.regression import *
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf

def get_dataframe(url, starttime, endtime, symbol='BTCUSDT', interval='1h', limit='1000'):
    params = {"symbol": symbol, 'interval': interval, 'startTime': starttime, 'endTime': endtime, 'limit': '1000'}
    return pd.DataFrame(json.loads(requests.get(url, params=params).text))

def get_historical_data():
    url = "https://api.binance.com/api/v3/klines"
    hist_df = pd.DataFrame()
    # get historical BTC data from past year
    for i in range(12, -1, -1):
        starttime = str(int((dt.datetime.today() - relativedelta(months=i+1)).timestamp() * 1000))
        endtime = str(int((dt.datetime.today() - relativedelta(months=i)).timestamp() * 1000))
        hist_df = pd.concat([hist_df, pd.DataFrame(get_dataframe(url, starttime, endtime))])
    # drop everything but opening prices
    hist_np = hist_df.iloc[:, 1].to_numpy().astype(float)
    plt.plot(range(len(hist_np)), hist_np)
    plt.title('original data')
    plt.show()
    # standardize data
    #hist_np = (hist_np - np.mean(hist_np)) / np.std(hist_np)
    plt.plot(range(len(hist_np)), hist_np)
    plt.title('standardized data')
    plt.show()

    return hist_np


def get_best_fitting_model(hist_np):


    train, test = np.split(hist_np, [int(0.67 * len(hist_np))])

    plt.plot(range(len(train)), train)
    plt.title('train set')
    plt.show()
    plt.plot(range(len(test)), test)
    plt.title('test set')
    plt.show()


    #columns = ['Series', 'Open_price']
    columns = ['Open_price']

    train, test = pd.DataFrame(train, columns=columns), pd.DataFrame(test, columns=columns)
    s = setup(data=train, test_data=test, target='Open_price', fold_strategy='timeseries', #numeric_features=['Series'],
              fold=3, transform_target=True, session_id=123)
    best = compare_models(sort='MSE')
    #prediction_holdout = predict_model(best)
    predictions = predict_model(best, data=pd.DataFrame(hist_np, columns=columns))#.iloc[:, 1]
    plt.plot(range(len(predictions)), predictions)
    plt.show()


def ARIMA_test(hist_np):
    hist_df = pd.DataFrame(hist_np, columns=['open_prices'])
    hist_df['series'] = range(len(hist_np))
    #print(hist_df.head())
    split = int(len(hist_df) * 0.85)
    train = hist_df[:split]
    test = hist_df[split:]

    print(train.head())
    plt.plot(range(len(train['open_prices'])),train['open_prices'])
    plt.title('train data')
    plt.show()
    plt.plot(range(len(test['open_prices'])),test['open_prices'])
    plt.title('test data')
    plt.show()
    
    #train, test = np.split(hist_np, [int(0.85 * len(hist_np))])
    #train_df, test_df = pd.DataFrame(train), pd.DataFrame(test)
    """
    model = ARIMA(train, order=(1, 1, 2))
    model_fit = model.fit(disp=0)
    model_fit.plot_predict(dynamic=False)
    plt.show()
    
    ARIMAmodel = ARIMA(train, order=(1, 1, 2))
    ARIMAmodel = ARIMAmodel.fit()
    pred = ARIMAmodel.get_forecast(len(test_df.index))
    pred_df = pd.DataFrame(pred.conf_int(alpha=0.05))
    print(pred_df.head())
    pred_df["Predictions"] = ARIMAmodel.predict(start=pred_df.index[0], end=pred_df.index[-1])
    pred_df.index = test_df.index
    pred_out = pred_df["Predictions"]
    plt.plot(pred_out, color='Red', label='ARIMA Predictions')
    plt.legend()
    plt.title('prediction data')
    plt.show()
    """

    model = ARIMA(train['open_prices'], order=(3, 2, 3))
    fitted = model.fit()
    print(fitted.summary())
    # Forecast
    fc = fitted.forecast(len(hist_df) - split, alpha=0.15)
    print(fc)
    # Make as pandas series
    #fc_series = pd.Series(fc, index=train.index)
    #lower_series = pd.Series(conf[:, 0], index=test.index)
    #upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    #plt.plot(train['open_prices'], label='training')
    #plt.plot(hist_df['open_prices'], label='actual')
    plt.plot(fc, label='forecast')
    #plt.fill_between(lower_series.index, lower_series, upper_series,
    #                 color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

    #arma_rmse = np.sqrt(mean_squared_error(test, pred_df["Predictions"]))
    #print("RMSE: ", arma_rmse)


def main():
    hist_np = get_historical_data()
    ARIMA_test(hist_np)
    #get_best_fitting_model(hist_np)



if __name__ == '__main__':
    main()