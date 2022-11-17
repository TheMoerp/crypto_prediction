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
    df = pd.DataFrame()
    for i in range(12, -1, -1):
        starttime = str(int((dt.datetime.today() - relativedelta(months=i+1)).timestamp() * 1000))
        endtime = str(int((dt.datetime.today() - relativedelta(months=i)).timestamp() * 1000))
        pd.concat((get_dataframe(url, starttime, endtime))

    #starttime = str(int((dt.datetime.today() - relativedelta(months=1)).timestamp() * 1000))
    #endtime = str(int((dt.datetime.today() - relativedelta(months=0)).timestamp() * 1000))
    #df = get_dataframe(url, starttime, endtime)
    df.head()
    #np_arr = df.iloc[:, 1].to_numpy().astype(np.float)
    #np_arr = (np_arr - np.mean(np_arr)) / np.std(np_arr)
    #print(np_arr)


if __name__ == '__main__':
    get_historical_data()