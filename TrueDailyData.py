from datetime import datetime,timedelta
import pandas_datareader as pdata
import numpy as np
end_date = datetime.now().date()
start_date = datetime.now().date() - timedelta(days=3)
filename = "/home/ian/SP500.csv"


def get_stock_change(symbol, start, end):
    _apiKey = "602e6966866fa8667b139219e0592d809c2a6fa0"
    stock_data = pdata.get_data_tiingo(symbols=symbol.upper(), start=start, end=end, api_key=_apiKey).as_matrix()[:, 0]
    return [symbol.upper(), ((stock_data[-1]-stock_data[-2])/stock_data[-2])*100]


def get_stock_list(file, start, end):
    start = start
    end = end

    stocks = np.genfromtxt(file, dtype=np.str, delimiter=',')
    out = []

    for sym in stocks:
        try:
            out.append(get_stock_change(sym, start, end))
        except (KeyError, TypeError) as e:
            print(sym)
            out.append([sym, 0.0])

    return np.asarray(out)


stocks = get_stock_list(filename, start_date, end_date)
np.savetxt("/home/ian/stock_report_true.csv", stocks, fmt="%s", delimiter=',')
