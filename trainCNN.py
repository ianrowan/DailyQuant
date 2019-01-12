import tensorflow as tf
import numpy as np
from DataBuilder import DataBuilder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class trainCnn:

    def __init__(self, symbol, categories, use_index=True, device_name="/GPU:0"):
        self.symbol = symbol
        self.categories = categories
        self.device_name = device_name
        self.use_index = use_index
        if use_index:
            self.channels = 2

    def build_network(self, input_placeholder, keep):
        def weight_var(shape):
            init = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(init)

        def bias_var(shape):
            init = tf.constant(0.5, shape=shape)
            return tf.Variable(init)

        def conv2d(x, W, stride):
            return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        def conv_layer(input_, shape, str=1, pool=False):
            with tf.name_scope('Weights'):
                w = weight_var(shape)
            with tf.name_scope('Bias'):
                b = bias_var([shape[3]])

            h_conv = tf.nn.relu(conv2d(input_, w, stride=str) + b)

            if pool == True:
                return max_pool_2x2(h_conv)
            return h_conv
        with tf.device(self.device_name):
            #Layer 1 - 20 x 5 x 1
            conv1 = conv_layer(input_placeholder, [5, 5, self.channels, 12], pool=False)
            #Layer 2 - 10 x 3 x 12
            conv2 = conv_layer(conv1, [5, 5, 12, 24], pool=False)
            #Layer 3 - 5 x 2 x 24
            conv3 = conv_layer(conv2, [5, 5, 24, 36], pool=True)

            conv4 = conv_layer(conv3, [5, 5, 36, 64], pool=True)
            #Layer 4 - 3 x 1 x 36
            #Fully connected 1
            w1 = weight_var([5*2*64, 960])
            b1 = bias_var([960])
            l1 = tf.reshape(conv4, [-1, 5*2*64])
            fc1 = tf.nn.relu(tf.matmul(l1, w1) + b1)
            #Fully Connected 2
            w2 = weight_var([960, 960])
            b2 = bias_var([960])
            fc2 = tf.nn.relu(tf.matmul(fc1, w2) + b2)
            fc2 = tf.nn.dropout(fc2, keep_prob=keep)
            #Fully Connected Output
            w3 = weight_var([960, self.categories])
            b3 = bias_var([self.categories])
            y_pred = tf.matmul(fc2, w3) + b3
        return y_pred

    def train_predict_new(self, steps, learn_rate):

        def next_batch(num, data, labels):

            idx = np.arange(0, len(data))
            np.random.shuffle(idx)
            idx = idx[:num]
            data_shuffle = [data[i] for i in idx]
            labels_shuffle = [labels[i] for i in idx]

            return np.asarray(data_shuffle), np.asarray(labels_shuffle)

        db = DataBuilder(self.symbol, 5, use_index=self.use_index, scale=True, num_cats=self.categories)
        data = db.get_data()

        x_in = tf.placeholder(tf.float32, shape=[None, 20, 5, self.channels])
        y_exp = tf.placeholder(tf.float32, shape=[None, self.categories])
        keep = tf.placeholder(tf.float32)
        if not self.use_index:
            x = np.reshape(data[0], newshape=[-1, 20, 5, 1])
        else:
            x = data[0]
        y = np.squeeze(np.eye(self.categories)[data[1].astype(np.int).reshape(-1)])

        x_, x_val, y_, y_val = train_test_split(x, y, test_size=.00)
        y_pred = self.build_network(x_in, keep)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_exp, logits=y_pred))
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

        #saver = tf.train.Saver()
        #train_loss = []
        #test_loss = []
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            tf.logging.set_verbosity(tf.logging.ERROR)
            for i in range(steps):
                batch = next_batch(32, x_, y_)
                feed = {x_in: batch[0], y_exp: batch[1], keep: 0.7}
                '''if i % 10 == 0:
                    feed_actual = {x_in: batch[0], y_exp: batch[1], keep: 1.0}
                    feed_val = {x_in: x_val, y_exp: y_val, keep: 1.0}
                    l1 = loss.eval(feed_dict=feed_actual)
                    l2 = loss.eval(feed_dict=feed_val)
                    if i%100 == 0:
                        print('========================SUMMARY REPORT=============================')
                        print('step %d, train loss: %g' % (i, l1))
                        print('Test loss: %g' % (l2))
                    train_loss.append(l1)
                    test_loss.append(l2)'''
                train_step.run(feed_dict=feed)
                #if i == steps -1:
                    #yy = y_pred.eval(feed_dict={x_in: x_val[:2], keep:1.0})
                    #y_out = yy[1]
                    #print(yy[1])
                    #print([np.exp(p-max(y_out))/sum([np.exp(j -max(y_out)) for j in y_out]) for p in y_out])
                    #print(y_val[1])
            x_new = db.get_recent_input()
            if not self.use_index:
                x_new = np.reshape(db.get_recent_input(), newshape=[-1, 20, 5, 1])
            yo = y_pred.eval(feed_dict={x_in: x_new, keep: 1.0})
            sess.close()
        '''plt.plot(train_loss)
        plt.plot(test_loss)
        plt.show()'''
        ranges = db.ranges
        gc.collect()
        del db, sess, y_pred, data
        return yo, ranges
        #, test_loss[-1]


#print(trainCnn("AMZN", 5).train_predict_new(1500, 1e-4))
'''
params = [[1500, 1e-4], [1500, 1e-3], [2000, 1e-4], [3000, 1e-4]]

losses = []
for i in params:
    losses.append(trainCnn("AMZN", 5).train_predict_new(i[0], i[1])[2])

print(losses)
'''

