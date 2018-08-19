import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os
import sys
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
import sklearn.cluster.k_means_
from sklearn.utils.extmath import row_norms, squared_norm
from numpy.random import RandomState
from sklearn.cluster import AgglomerativeClustering
import time
import heapq
import datetime
import copy

def k_center_vector_fp32(weight_vector, n_clusters, verbosity=0, seed=int(time.time()), gpu_id=7, name='', epoch=0, labels_new=[], centers_new=[]):

    if n_clusters == 1:

        mean_sample = np.mean(weight_vector, axis=0)

        weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))

        return weight_vector, [], []

    elif weight_vector.shape[0] == n_clusters:

        return weight_vector, [], []

    # elif weight_vector.shape[1] == 1:
    #
    # 	return k_means_vector(weight_vector, n_clusters, seed=seed)

    else:
        weight_vector, min_i = smallest_to_zero(weight_vector)
        if epoch > 0:
            centers = center(weight_vector, labels_new, centers_new)
            labels = labels_new
        else:
            centers, labels = K_cluster(weight_vector, n_clusters, min_i)
            centers[0] = center(weight_vector, labels, centers)[0]
        # if weight_vector.shape[1] == 1:
        # 	plot(weight_vector, labels, centers, str(n_clusters) + '_k_center_' + str(layer))
        # 	pass
        deleted = []
        # centers, deleted = k_center_renew_zero(centers, labels)
        # for i in centers:
        # 	if i[0] == 0 and i[1] == 0 and i[2] == 0:
        # 		continue
        # 	else:
        # 		print(i)

        # centers, _ = k_center_renew_zero(centers, labels, int(n_clusters/8))

        # stat = statistic(centers, labels, weight_vector)
        # f = open('./stat/layer_' + str(layer) + '.txt', 'w')

        # for s in stat:
        # print(s, ': \t', stat[s][0], '; \t', stat[s][1][0], '; \t', stat[s][2][0])
        # print(str(layer) + ': \t' + str(centers[s]) + ': \t' + str(stat[s][0] / weight_vector.shape[0]))
        # if s == 0:
        # 	break
        # f.close()

        # print(memory_huff(labels, weight_vector.shape[0], centers, weight_vector, deleted))
        # if epoch % 36 == 0 and epoch != 0:
        # plot(weight_vector, labels, centers, str(n_clusters) + '_k_center_' + str(layer))

        weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
        for v in range(weight_vector.shape[0]):
            weight_vector_compress[v, :] = centers[labels[v], :]
        # weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
        return weight_vector_compress, labels, centers


def statistic(centers, labels, vectors):
    times = dict(zip(*np.unique(labels, return_counts=True)))
    for i in times:
        time = times[i]
        var = 0
        sum = vectors[0] * 0
        num = 0
        for index, label in enumerate(labels):
            if label == i:
                num += 1
                sum += vectors[index]
        mean = sum / num
        for index, label in enumerate(labels):
            if label == i:
                var += paired_distances(vectors[index].reshape(1, -1), mean.reshape(1, -1))
        var /= num
        norm = paired_distances(centers[i].reshape(1, -1), (centers[i]*0).reshape(1, -1))
        times[i] = [time, var, norm]
    return times


def K_initialize(vectors, k, zero_ini):
    n = vectors.shape[0]
    d = vectors.shape[1]
    labels_dist = np.zeros((n, 2))
    centers = np.zeros((k, d))
    # np.random.seed(0)
    # ini = np.random.randint(n)
    ini = zero_ini
    for i, item in enumerate(labels_dist):
        item[0] = 0
        item[1] = paired_distances(vectors[ini].reshape((1, -1)), vectors[i].reshape((1, -1)))[0]
    centers[0] = vectors[ini]
    return labels_dist, centers


def K_cluster(vectors, k, zero_ini):
    labels_dist, centers = K_initialize(vectors, k, zero_ini)
    for i in range(1, k):
        D_index = np.argmax(labels_dist[:, 1])
        D = labels_dist[D_index, 1]
        labels_dist[D_index, 1] = 0
        labels_dist[D_index, 0] = i
        centers[i] = vectors[D_index]
        centers_repeat = np.repeat(centers[i].reshape(1, -1), vectors.shape[0], axis=0)
        dist_matrix = paired_distances(vectors, centers_repeat)
        dist_matrix[D_index] = 1
        sign_matrix = dist_matrix.__le__(labels_dist[:, 1])
        labels_dist[:, 0] = ~sign_matrix * labels_dist[:, 0] + sign_matrix * i
        labels_dist[:, 1] = ~sign_matrix * labels_dist[:, 1] + sign_matrix * dist_matrix
    return centers, labels_dist[:, 0].astype(int)


def center(vectors, labels, center):
    centers = copy.deepcopy(center)
    centers *= 0
    dict_t = {}
    for l, v in zip(labels, vectors):
        centers[l] += v
        if l in dict_t.keys():
            dict_t[l] += 1
        else:
            dict_t[l] = 1

    for i, c in enumerate(centers):
        c /= dict_t[i]

    return centers


def center_min(vectors, labels, center):
    centers = copy.deepcopy(center)
    centers *= 0
    dict_t = {}
    for l, v in zip(labels, vectors):
        centers[l] += v
        if l in dict_t.keys():
            dict_t[l] += 1
        else:
            dict_t[l] = 1

    for i, c in enumerate(centers):
        c /= dict_t[i]

    for l, v in zip(labels[1:], vectors[1:]):
        centers[l] += v
        if l in dict_t.keys():
            if paired_distances(v.reshape(1, -1), (v * 0).reshape(1, -1)) <= paired_distances(dict_t[l].reshape(1, -1), (v * 0).reshape(1, -1)):
                dict_t[l] = v
        else:
            dict_t[l] = v

    for i, c in enumerate(centers):
        centers[i] = dict_t[i]

    return centers


def smallest_to_zero(vectors):
    min_i = 0
    min_v = vectors[0]
    vectors_mean = np.mean(vectors, axis=0)
    for i, v in enumerate(vectors):
        # if paired_distances(v.reshape(1, -1), (v * 0).reshape(1, -1)) <= paired_distances(min_v.reshape(1, -1), (min_v * 0).reshape(1, -1)):
        # 	min_v = v
        # 	min_i = i
        if paired_distances(v.reshape(1, -1), vectors_mean.reshape(1, -1)) <= paired_distances(min_v.reshape(1, -1), vectors_mean.reshape(1, -1)):
            min_v = v
            min_i = i
    vectors[min_i] *= 0		# can use this initial value(the smallest value)
    return vectors, min_i


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass