import concurrent.futures
import os
import numpy as np
import objgraph
from datetime import datetime, timedelta
from trainCNN import TrainSingle, TrainOverall

dir_path = os.path.dirname(os.path.realpath(__file__))


class PredictList:

    def __init__(self, stock_list, device_name="/GPU:0"):
        self.stock_list = stock_list
        self.device = device_name

    def predict_one(self, symbol, categories):
        model_path = dir_path + "/TensorFlow/models/Full_model.ckpt"

        if not os.path.exists(model_path) and os.path.getmtime(model_path) < (datetime.now() - timedelta(days=30)):
            TrainOverall(self.stock_list, categories, True).train_full_network(30000, 1e-4, 4000, 0.5, model_path)

        prediction = TrainSingle(symbol, categories, self.device).train_predict_new(450, 1e-3, model_path)
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


if __name__ == '__main__':

    def create_list(stock_list, device):
        return PredictList(stock_list, device).log_builder()

    stocks = np.genfromtxt(dir_path + "/SP500.csv", dtype=np.str, delimiter=',')
    threads = 7

    stocks = [stocks[(i - 1) * int((504 / threads)): i * int((504 / threads))] for i in range(1, threads + 1)]
    gpus = ["/GPU:0"for i in range(threads)]

    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
        results = [x for x in executor.map(create_list, stocks, gpus)]
    output = np.concatenate(tuple([results[i] for i in range(threads)]), axis=0)

    #output = PredictList(stocks).log_builder()

    print(output)

    # noinspection PyTypeChecker
    np.savetxt(dir_path + "/stock_report.csv", output, fmt="%s", delimiter=',')