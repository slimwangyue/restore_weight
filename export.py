from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import main

import numpy as np

labels_all, centers_all, index_all, appendix_all, shapes_all, freq_all, k_all, name_all = main.get_meta()


restore_module = tf.load_op_library('/home/yw68/tensorflow/tensorflow/core/user_ops/restore_weight.so')


def full_layer(input_feature, n, index_len, center_shape, out_depth):
    index = tf.get_variable(n + "index", shape=index_len, dtype=tf.int8)
    print("index_shape:", index.get_shape())

    centers = tf.get_variable(n + "centers", shape=(center_shape[0], center_shape[1]), dtype=tf.float32)
    print("centers_shape:", centers.get_shape())

    shape = tf.get_variable(n + "shape", shape=4, dtype=tf.int32)
    print("shape_shape:", shape.get_shape())

    freq = tf.get_variable(n + "freq", shape=center_shape[0], dtype=tf.int32)
    print("freq_shape:", freq.get_shape())
    # r_ = tf.matmul(i,W)
    # print("r_:", r_.get_shape())
    # r = r_ + b
    filter_res = restore_module.restore_weight(index, centers, shape, freq)
    result = tf.nn.conv2d(input=input_feature, filter=filter_res, strides=[1, 1, 1, 1], padding='SAME')
    b = tf.Variable(tf.random_normal([out_depth]))
    result = tf.nn.bias_add(result, b)
    result = tf.nn.relu6(result)
    result = tf.layers.max_pooling2d(result, 2, 2)

    return result


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
num_input = 784     # MNIST data input (img shape: 28*28)
num_classes = 10    # MNIST total classes (0-9 digits)

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name='conv1_w'),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='conv2_w'),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


#   our customized model
tmp_name = ''
for i in labels_all.keys():
    if 'conv2_w' in i:
        tmp_name = i

activation = tf.reshape(X, shape=[-1, 28, 28, 1])
# Convolution Layer
activation = conv2d(activation, weights['wc1'], biases['bc1'])
# Max Pooling (down-sampling)
activation = maxpool2d(activation, k=2)
activation = full_layer(activation, '1', appendix_all[tmp_name], centers_all[tmp_name].shape, 64)
# Fully connected layer
# Reshape conv2 output to fit fully connected layer input
activation = tf.reshape(activation, [-1, weights['wd1'].get_shape().as_list()[0]])
activation = tf.add(tf.matmul(activation, weights['wd1']), biases['bd1'])
activation = tf.nn.relu(activation)

# Output, class prediction
logits = tf.add(tf.matmul(activation, weights['out']), biases['out'])

prediction = tf.nn.softmax(logits)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    memory = 0
    use_compression = True
    memory_weight = 0
    memory_weight_log = 0

    for v in tf.trainable_variables():  # TODO: just one layer been clustered!!!
        if 'index' in v.name:

            index = sess.run(v)
            tmp_name = ''
            for i in labels_all.keys():
                if 'conv2_w' in i:
                    tmp_name = i

            for i, item in enumerate(index):
                string_tmp = index_all[tmp_name][i * 8: i * 8 + 8]
                index[i] = int(string_tmp, 2)   # converted automatically

            sess.run(v.assign(index))

            print('==================assigning index done==================')

        if 'centers' in v.name:

            centers = sess.run(v)
            tmp_name = ''
            for i in centers_all.keys():
                if 'conv2_w' in i:
                    tmp_name = i

            sess.run(v.assign(centers_all[tmp_name]))

            print('==================assigning centers done==================')

        if 'shape' in v.name:

            shape = sess.run(v)
            tmp_name = ''
            for i in shapes_all.keys():
                if 'conv2_w' in i:
                    tmp_name = i

            for i in range(len(shape)):
                shape[i] = shapes_all[tmp_name][i]
            sess.run(v.assign(shape))

            print('==================assigning shape done==================')

        if 'freq' in v.name:

            freq = sess.run(v)
            tmp_name = ''
            for i in freq_all.keys():
                if 'conv2_w' in i:
                    tmp_name = i
            for i in range(len(freq_all[tmp_name].keys())):
                freq[i] = freq_all[tmp_name][str(i)]
            sess.run(v.assign(freq))

            print('==================assigning freq done==================')

    print("Assigning Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[:256], Y: mnist.test.labels[:256]}))