from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import numpy as np

import kcenter

import huffman

labels_all = {}
centers_all = {}
index_all = {}  # num % 8 = 0
appendix_all = {}
shapes_all = {}
freq_all = {}
k_all = {}
name_all = {}   # layer_num -> name


def cluster_conv(weight, n_clusters, seed, name, epoch):
    from sklearn.metrics import mean_squared_error
    # weight: cuda tensor
    filters_num = weight.shape[0]
    filters_channel = weight.shape[1]
    filters_size = weight.shape[2]

    weight_vector = weight.reshape(-1, filters_size)

    weight_vector_clustered, labels_all[name], centers_all[name] = kcenter.k_center_vector_fp32(weight_vector.astype('float32'), n_clusters, verbosity=0, seed=seed, gpu_id=0, labels_new=labels_all[name], centers_new=centers_all[name], epoch=epoch)

    unique, counts = np.unique(labels_all[name], return_counts=True)
    times = {}
    for i in range(n_clusters):
        times[str(i)] = counts[i]
    h = huffman.HuffmanCoding()
    huffman_dict = h.compress(times)
    huffman_code = ''
    for i in labels_all[name]:
        huffman_code += huffman_dict[str(i)]
    appendix_len = 8 - len(huffman_code) % 8
    for i in range(appendix_len % 8):
        huffman_code += '0'
    appendix_string = ''
    for i in range(8):
        if i == appendix_len % 8:
            appendix_string += '1'
        else:
            appendix_string += '0'
    huffman_code += appendix_string


    index_all[name] = huffman_code
    appendix_all[name] = int(len(huffman_code) / 8)
    freq_all[name] = times

    all_count = labels_all[name].shape[0]
    prob = counts / all_count
    huffman_length = np.sum([- p * np.log2(p) for p in prob])

    print('huffman_length: {}'.format(huffman_length))
    print('log2: {}'.format(np.log2(n_clusters)))
    print('prob_max: ', np.max(prob))
    print('abs mean: ', np.mean(np.abs(weight_vector), axis=0))
    print('plot begins', name)

    # if filters_size == 3:
    #     plot(weight_vector, labels_all[name], centers_all[name], name, epoch, n_clusters, np.max(prob))

    weight_cube_clustered = weight_vector_clustered.reshape(filters_num, filters_channel,
                                                            filters_size, -1)

    mse = mean_squared_error(weight_vector, weight_vector_clustered)


    weight_compress = weight_cube_clustered.astype('float32')

    return weight_compress, mse, huffman_length


def get_meta():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Training Parameters
    learning_rate = 0.001
    num_steps = 20
    batch_size = 128
    display_step = 10

    # Network Parameters
    num_input = 784 # MNIST data input (img shape: 28*28)
    num_classes = 10 # MNIST total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to keep units

    # tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


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


    # Create model
    def conv_net(x, weights, biases, dropout):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
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

    # Construct model
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)


    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training
        for step in range(1, num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))


        memory = 0
        use_compression = True
        memory_weight = 0
        memory_weight_log = 0

        # Clustering
        for v in tf.trainable_variables():
            if 'conv2_w' in v.name:
                if v.name not in labels_all.keys():
                    labels_all[v.name] = []
                    centers_all[v.name] = []

                memory += np.prod(sess.run(v).shape)
                print("weights.name: {}".format(v.name))
                print("weights.shape: {}".format(sess.run(v).shape))
                if use_compression:
                    weights = sess.run(v)
                    shapes_all[v.name] = weights.shape

                    weights = np.transpose(weights, (3, 2, 1, 0))
                    shape = weights.shape
                    n, c, w = shape[0], shape[1], shape[2]
                    k = 300
                    k_all[v.name] = k
                    # skip the first layer
                    # if v.name == 'InceptionV1/Conv2d_1a_7x7/weights:0':
                    #     k = n * c * w

                    weight_clustered, mse, huffman_length = cluster_conv(weights, k, 0, v.name, 0)

                    samples_num = n * c * w
                    feature_dim = w
                    # weight_reshape = weight_clustered.reshape(-1, w)
                    # unique, counts = np.unique(weight_reshape, return_counts=True, axis=0)
                    # all_count = weight_reshape.shape[0]
                    # prob = counts / all_count
                    # huffman_length = np.sum([- p * np.log2(p) for p in prob])

                    memory_weight += k * feature_dim * 32 + samples_num * huffman_length
                    memory_weight_log += k * feature_dim * 32 + samples_num * np.log2(k)

                    # self.memory_weight = cluster_list_conv[i] * feature_dim * self.bitwidth_org + samples_num * np.log2(cluster_list_conv[i])

                    weight_clustered = np.transpose(weight_clustered, (3, 2, 1, 0))
                    sess.run(v.assign(weight_clustered))
                    print("weight_clustered shape: {}".format(weight_clustered.shape))
                    print("mse: {}".format(mse))

                print('==================cluster done==================')

        print("Optimization Finished!")

        # Calculate accuracy for 256 MNIST test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                          Y: mnist.test.labels[:256],
                                          keep_prob: 1.0}))

    return labels_all, centers_all, index_all, appendix_all, shapes_all, freq_all, k_all, name_all






