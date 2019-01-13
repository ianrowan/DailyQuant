import tensorflow as tf
import numpy as np
from DataBuilder import DataBuilder
import tqdm
from sklearn.model_selection import train_test_split
import gc
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_path = os.path.dirname(os.path.realpath(__file__))


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


class TrainOverall:

    def __init__(self, stock_list, categories, use_index):
        self.stocks = stock_list
        self.categories = categories
        self.use_index = use_index
        if use_index:
            self.channels = 2

    def _build_dataset(self):
        data_in = np.zeros(shape=[len(self.stocks)*5*252, 20, 5, 2])
        data_out = np.zeros(shape=[len(self.stocks)*5*252, self.categories])
        i = 0
        for stock in self.stocks:
            try:
                data = DataBuilder(stock, 5, scale_single=True).get_data()
                data_in[i:i+len(data[0]), :, :, :] = data[0]
                data_out[i:i + len(data[1]), :] = np.squeeze(np.eye(self.categories)[data[1].astype(np.int).reshape(-1)])
                i += len(data[1])
            except (TypeError, ValueError):
                continue

        return np.delete(data_in, np.s_[i:], axis=0), np.delete(data_out, np.s_[i:], axis=0)

    def _build_network(self, input_placeholder, keep):
        def weight_var(shape):
            init = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(init)

        def bias_var(shape):
            init = tf.constant(0.01, shape=shape)
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

            h_conv = tf.nn.relu(tf.layers.batch_normalization(conv2d(input_, w, stride=str) + b))

            if pool:
                return max_pool_2x2(h_conv)
            return h_conv

        # Layer 1 - 20 x 5 x 2
        conv1 = conv_layer(input_placeholder, [3, 3, self.channels, 12], pool=False)
        # Layer 2 - 20 x 5 x 12
        conv2 = conv_layer(conv1, [3, 3, 12, 24], pool=False)
        # Layer 3 - 10 x 3 x 24
        conv3 = conv_layer(conv2, [3, 3, 24, 36], pool=True)
        # Layer 4 - 5 x 2 x 36
        conv4 = conv_layer(conv3, [3, 3, 36, 64], pool=True)
        with tf.variable_scope("FcTrain"):
            # Fully connected 1
            w1 = weight_var([5*2*64, 120])
            b1 = bias_var([120])
            l1 = tf.reshape(conv4, [-1, 5*2*64])
            fc1 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.matmul(l1, w1) + b1))
            fc1 = tf.nn.dropout(fc1, keep_prob=keep)
            # Fully Connected 2
            w2 = weight_var([120, 60])
            b2 = bias_var([60])
            fc2 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.matmul(fc1, w2) + b2))
            fc2 = tf.nn.dropout(fc2, keep_prob=keep)
            # Fully Connected Output
            w3 = weight_var([60, self.categories])
            b3 = bias_var([self.categories])
        y_pred = tf.matmul(fc2, w3) + b3
        tf.identity(y_pred, "y_pred")
        return y_pred

    def train_full_network(self, steps, learn_rate, batch_size, keep_prob, model_path):

        '''
        data = self._build_dataset()
        try:
            np.save(dir_path + "/data_array0", data[0])
            np.save(dir_path + "/data_array1", data[1])
        except ValueError as e:
            print(e)
        '''
        data = [np.load(dir_path + "/data_array0.npy"), np.load(dir_path + "/data_array1.npy")]
        print("data size {}".format(str(len(data[0]))))
        x_in = tf.placeholder(tf.float32, shape=[None, 20, 5, self.channels], name="x")
        print(x_in)
        y_exp = tf.placeholder(tf.float32, shape=[None, self.categories], name="y")
        keep = tf.placeholder(tf.float32, name="keep")
        if not self.use_index:
            x = np.reshape(data[0], newshape=[-1, 20, 5, 1])
        else:
            x = data[0]
        x_, x_val, y_, y_val = train_test_split(x, data[1], test_size=.20)

        y_pred = self._build_network(x_in, keep_prob)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_exp, logits=y_pred))
        print(loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
            train_step_single = tf.train.AdamOptimizer(learn_rate).minimize(loss,
                                                                            var_list=
                                                                            [var for var in tf.trainable_variables()
                                                                             if var.name.startswith('FcTrain')])
        print(train_step_single)
        loss_graph = tf.summary.scalar("T_Loss", loss)
        loss_graphv = tf.summary.scalar("Val_Loss", loss)

        saver = tf.train.Saver(tf.all_variables())
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            write = tf.summary.FileWriter(dir_path + "/TensorFlow/TensorBoard", sess.graph)
            try:
                for i in tqdm.tqdm_gui(range(steps)):
                    batch = next_batch(batch_size, x_, y_)
                    feed = {x_in: batch[0], y_exp: batch[1], keep: 1.0}
                    if i % 500 == 0:
                        val_batch = next_batch(5000, x_val, y_val)
                        feed_val = {x_in: val_batch[0], y_exp: val_batch[1], keep: 1.0}
                        l1 = loss.eval(feed_dict=feed)
                        l2 = loss.eval(feed_dict=feed_val)
                        gl = loss_graph.eval(feed_dict={loss: l1})
                        gv = loss_graphv.eval(feed_dict={loss: l2})
                        print('========================SUMMARY REPORT=============================')
                        print('Epoch: {}'.format(str((batch_size*i)/len(x))))
                        print('Step: %d, train loss: %g' % (i, l1))
                        print('Test loss: %g' % l2)
                        write.add_summary(gl, i)
                        write.add_summary(gv, i)
                    feed[keep] = keep_prob
                    train_step.run(feed_dict=feed)
                    if i % (steps * 0.1) == 0:
                        saver.save(sess, model_path)
                        print("Checkpoint saved at {}".format(str(i)))
            except KeyboardInterrupt:
                pass
            saver.save(sess, model_path)


class TrainSingle:

    def __init__(self, symbol, categories, use_index=True, device_name="/GPU:0"):
        self.symbol = symbol
        self.categories = categories
        self.device_name = device_name
        self.use_index = use_index
        if use_index:
            self.channels = 2

    def _get_data(self):
        db = DataBuilder(self.symbol, 20, use_index=self.use_index, scale_single=True, num_cats=self.categories)
        data = db.get_data()

        if not self.use_index:
            x = np.reshape(data[0], newshape=[-1, 20, 5, 1])
        else:
            x = data[0]
        y = np.squeeze(np.eye(self.categories)[data[1].astype(np.int).reshape(-1)])

        x_new = db.get_recent_input()
        if not self.use_index:
            x_new = np.reshape(db.get_recent_input(), newshape=[-1, 20, 5, 1])
        return x, y, x_new, db.ranges

    def train_predict_new(self, steps, model_path):
        data = self._get_data()
        x_, x_val, y_, y_val = train_test_split(data[0], data[1], test_size=.04)

        meta = model_path + ".meta"
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.train.import_meta_graph(meta).restore(sess, model_path)
            with tf.device(self.device_name):
                for i in range(steps):
                    batch = next_batch(500, x_, y_)
                    feed = {"x:0": batch[0], "y:0": batch[1], "keep:0": 0.38}
                    sess.run("Adam_1", feed_dict=feed)
                    '''if i % 10 == 0:
                        feed["keep:0"] = 1.0
                        print(("==============Step: {}==============".format(str(i))))
                        print("Loss: {}".format(sess.run("Mean:0", feed_dict={"x:0": x_val, "y:0": y_val, "keep:0": 1.00})))
                    '''
                yo = sess.run("add_4:0", feed_dict={"x:0": data[2], "keep:0": 1.0})
                sess.close()

        ranges = data[3]
        gc.collect()
        del sess, data
        return yo, ranges



