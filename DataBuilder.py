import numpy as np
import pandas_datareader.data as pdata
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


class DataBuilder:

    ranges = object()

    def __init__(self, symbol, years, use_index=True, categorize=True, scale_full=False, scale_single=False, num_cats=5):
        self.endDate = datetime.now().date() - timedelta(days=1)
        self.start = datetime(self.endDate.year - int(years), self.endDate.month, self.endDate.day) - timedelta(days=1)
        self._apiKey = "af08d3cdca987b9b5f0101ca0a63984ce6ab03d0"
        self.scale_single = scale_single
        self.stockData = pdata.get_data_tiingo(symbols=symbol.upper(), start=self.start, end=self.endDate,
                                               api_key=self._apiKey).as_matrix()[:, :5]
        self.stockData_out = np.asarray([x for x in self.stockData])
        data_points = len(self.stockData)
        print(str(data_points) + " records received for: " + symbol)
        if scale_full:
            scaler = MinMaxScaler()
            self.stockData = scaler.fit_transform(self.stockData)
        self.inputs = np.zeros(shape=[data_points - 20, 20, 5])
        self.outputs = np.zeros(shape=data_points - 20)
        self._build_raw()
        self.outputs_freeze = self.outputs
        if categorize:
            self.outputs = self.categorize_outputs(num_cats)
        if use_index:
            self.index_data = self.get_save_index(False)
            self.inputs = self._append_index(self.inputs, self.index_data[0])

    def _build_raw(self):
        for i in range(20, len(self.stockData)):
            self.inputs[i-20:i] = self.stockData[i-20:i] if not self.scale_single else MinMaxScaler().fit_transform(self.stockData[i-20:i])
            self.outputs[i - 20] = (self.stockData_out[i][0] - self.stockData_out[i - 1][0]) / self.stockData_out[i - 1][0]

    def adaptive_range(self, quantiles=4, show_dist=False):
        quantile_range = 1/quantiles
        quantiles = [quantile_range*i for i in range(quantiles+1)]
        quantile_ranges = np.quantile(self.outputs_freeze, quantiles)
        #print(quantiles)
        #print(quantile_ranges)
        if show_dist:
            plt.hist(self.outputs_freeze, bins=30)
            for quantile in quantile_ranges:
                plt.axvline(quantile, color='r')
            plt.show()
        return quantile_ranges

    def categorize_outputs(self, categories):
        self.ranges = self.adaptive_range(quantiles=categories)
        outputs = np.zeros(shape=len(self.stockData) - 20)
        for i, point in enumerate(self.outputs):
            for j in range(len(self.ranges)-1):
                if point >= self.ranges[j] and point < self.ranges[j+1]:
                    outputs[i] = j
                    break
        return outputs

    def get_data(self):

        return self.inputs, self.outputs

    def get_recent_input(self):
        out = self.stockData[-20:] if not self.scale_single else MinMaxScaler().fit_transform(self.stockData[-20:])
        out = np.reshape(out, newshape=[1, 20, 5])
        if self.index_data:
            out = self._append_index(out, np.reshape(self.index_data[1], newshape=[1, 20, 5]))
        return out

    def get_save_index(self, scale=True):
        if os.path.isfile(dir_path + "/index_data.npy"):
            index = np.load(dir_path + "/index_data.npy")
        else:
            index = pdata.get_data_tiingo(symbols="DIA", start=self.start, end=self.endDate,
                          api_key=self._apiKey).as_matrix()[:, :5]
            np.save(dir_path + "/home/ian/index_data", index)
        if scale:
            scaler = MinMaxScaler()
            index = scaler.fit_transform(index)
        index_array = np.zeros(shape=[len(index) - 20, 20, 5])
        recent_array = index[-20:] if not self.scale_single else MinMaxScaler().fit_transform(index[-20:])
        for i in range(20, len(index)):
            index_array[i - 20] = index[i - 20:i] if not self.scale_single else MinMaxScaler().fit_transform(index[i - 20:i])
        return index_array, recent_array

    def _append_index(self, inp, index):
        inputs = np.reshape(inp, (len(inp), 20, 5, 1))
        index = np.reshape(index[-len(self.inputs):], (len(inp), 20, 5, 1))
        return np.concatenate((inputs, index), axis=3)



'''
d = DataBuilder("AMZN",5, categorize=False)

print(d.adaptive_range(5, True))
print(d.ranges)
print(max(d.get_data()[1])*100)


x = DataBuilder("AMZN", 5).get_recent_input()
print(np.shape(x))
print(x[:,:,:,0])
'''

