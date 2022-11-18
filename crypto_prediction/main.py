import requests
import pandas as pd
import json
import matplotlib as plt
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta


def get_dataframe(url, starttime, endtime, symbol='BTCUSDT', interval='1h', limit='1000'):
    params = {"symbol": symbol, 'interval': interval, 'startTime': starttime, 'endTime': endtime, 'limit': '1000'}
    return pd.DataFrame(json.loads(requests.get(url, params=params).text))

def get_historical_data():
    url = "https://api.binance.com/api/v3/klines"
    hist_df = pd.DataFrame()
    for i in range(12, -1, -1):
        starttime = str(int((dt.datetime.today() - relativedelta(months=i+1)).timestamp() * 1000))
        endtime = str(int((dt.datetime.today() - relativedelta(months=i)).timestamp() * 1000))
        hist_df = pd.concat([hist_df, pd.DataFrame(get_dataframe(url, starttime, endtime))])

    hist_np = hist_df.iloc[:, 1].to_numpy().astype(float)
    #hist_np = (hist_np - np.mean(hist_np)) / np.std(hist_np)
    plt.plot(range(len(hist_np)), hist_np)
    plt.show()


if __name__ == '__main__':
    get_historical_data()