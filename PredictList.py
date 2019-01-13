import os
import numpy as np
from trainCNN import TrainSingle, TrainOverall
from datetime import datetime, timedelta
import objgraph
import concurrent.futures
dir_path = os.path.dirname(os.path.realpath(__file__))


class PredictList:

    def __init__(self, stock_list, device_name="/GPU:0"):
        self.stock_list = stock_list
        self.device = device_name

    def predict_one(self, symbol, categories):
        model_path = dir_path + "/TensorFlow/models/Full_model.ckpt"

        if not os.path.exists(model_path) and os.path.getmtime(model_path) < (datetime.now() - timedelta(days=30)):
            TrainOverall(self.stock_list, categories, True).train_full_network(30000, 1e-4, 4000, 0.5, model_path)

        prediction = TrainSingle(symbol, categories, self.device).train_predict_new(2000, model_path)
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
#TODO: FIX GPU Memory Leak!!

    stocks = np.genfromtxt(dir_path + "/SP500.csv", dtype=np.str, delimiter=',')

    stocks1 = stocks[:250]
    stocks2 = stocks[250:500]
    #stocks3 = stocks[300:400]
    #stocks4 = stocks[400:500]
    #stocks5 = stocks[400:500]
    #stocks6 = stocks[350:400]
    #stocks7 = stocks[400:450]
    #stocks8 = stocks[450:500]
    gpus = ["/GPU:0", "/GPU:0"]
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        results = [x for x in executor.map(create_list, [stocks1, stocks2], gpus)]
    output = np.concatenate((results[0], results[1]), axis=0)

    #output = PredictList(stocks).log_builder()

    print(output)

    np.savetxt(dir_path + "/stock_report.csv", output, fmt="%s", delimiter=',')

