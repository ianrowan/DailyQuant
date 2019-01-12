import numpy as np
from trainCNN import trainCnn
import sys
import objgraph
import concurrent.futures
import gc

class PredictList:

    def __init__(self, stock_list, device_name="/GPU:0"):
        self.stock_list = stock_list
        self.device = device_name

    def predict_one(self, symbol, categories):
        prediction = trainCnn(symbol, categories, self.device).train_predict_new(1800, 1e-4)
        return np.argmax(prediction[0][0]), prediction[1]

    def all_stocks(self):
        category = []
        price_range = []
        i = 0
        for stock in self.stock_list:
            try:
                pred = self.predict_one(stock, 5)
                category.append(pred[0])
                price_range.append([pred[1][pred[0]]*100, pred[1][pred[0]+1]*100])
                print(self.device + ": " + str(i))
                if i % 50 == 0:
                    objgraph.get_leaking_objects()
            except (TypeError, ValueError) as e:
                category.append(0)
                price_range.append([0.0, 0.0])
                print("Type Error for {}".format(stock))
                print(e)
            i += 1

        return category, price_range

    def log_builder(self):
        output = np.empty(shape=[len(self.stock_list), 4], dtype="U10")
        #output[0] = ["Symbol", "% Low", "% High", "Category"]
        get_preds = self.all_stocks()
        output[:, 0] = np.transpose(self.stock_list)
        output[:, [1, 2]] = get_preds[1]
        output[:, 3] = np.transpose(get_preds[0])
        return output



if __name__=='__main__':

    def create_list(stock_list, device):
        return PredictList(stock_list, device).log_builder()

    stocks = np.genfromtxt("/home/ian/SP500.csv", dtype=np.str, delimiter=',')
    stocks1 = stocks[:75]
    stocks2 = stocks[75:150]
    stocks3 = stocks[150:225]
    stocks4 = stocks[225:300]
    stocks5 = stocks[300:350]
    stocks6 = stocks[350:400]
    stocks7 = stocks[400:450]
    stocks8 = stocks[450:500]
    gpus = ["/GPU:0", "/GPU:0", "/GPU:0", "/GPU:0", "/GPU:1", "/GPU:1", "/GPU:1", "/GPU:1"]
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        results = [x for x in executor.map(create_list, [stocks1, stocks2, stocks3, stocks4, stocks5, stocks6, stocks7, stocks8], gpus)]
    output = np.concatenate((results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]), axis=0)
    '''
    output = PredictList(stocks[300:375]).log_builder()
    '''
    print(output)

    np.savetxt("/home/ian/stock_report.csv", output, fmt="%s", delimiter=',')

