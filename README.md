# DailyQuant
DailyQuant is a machine learning engine focused on predicting the next days stock changes for any stock on the market.

Daily Quant uses a Convolutional nerual network tot process "images" with 20x5x2 dimensions containg each of 20 subsequent day's HLOC and volume data on the first channel and the selected index(DJI, S&P, etc.)matching data on the 2nd channel.

The network uses a transfer learning approach in which the entire network is trained with ~600,000 stock records to train the convolutions to discern features and patterns between the stock data and the index whcih the network is tracking. When a single stock prediction is desired, a quick training of the fully connected layers based on that stock's specific data suffices for accurate results.
