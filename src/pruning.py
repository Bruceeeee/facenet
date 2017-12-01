"""Some function for pruning"""

import os
import numpy as np
import tensorflow as tf
import facenet


def get_masks(weights, percentile, layer_type=None):
    """ Use percentile to get weight mask"""
    weights_name = [tensor.name for tensor in tf.get_default_graph().get_operations()
                    if tensor.name.endswith('weights')]
    numpy_weights = np.array([weight.eval() for weight in weights])
    lower_thrs, upper_thrs = [], []
    for weight, name in zip(numpy_weights, weights_name):
        if layer_type is None:
            lower_thr = np.percentile(weight[weight != 0], percentile / 2.0)
            upper_thr = np.percentile(
                weight[weight != 0], 100 - percentile / 2.0)
        elif name in layer_type:
            lower_thr = np.percentile(weight[weight != 0], percentile / 2.0)
            upper_thr = np.percentile(
                weight[weight != 0], 100 - percentile / 2.0)
        else:
            lower_thr = upper_thr = 0
        lower_thrs.append(lower_thr)
        upper_thrs.append(upper_thr)
    masks = [(weight <= lower_thr) + (weight >= upper_thr)
             for weight, lower_thr, upper_thr in
             zip(numpy_weights, lower_thrs, upper_thrs)]
    return masks


def cal_pruning_rate(weights):
    nrof_zeros = np.sum(
        np.array([sum(weight.eval().ravel() == 0) for weight in weights]))
    total_w = np.sum(np.array([weight.eval().size for weight in weights]))
    print("The total number of weights is {} and {} zeros after pruning".format(
        total_w, nrof_zeros))
    rate = nrof_zeros * 1.0 / total_w
    print("The pruning rate is {:.3f}".format(rate))
    return rate, total_w, nrof_zeros


def apply_masks(weights, masks, layer_type=None):
    if layer_type is not None:
        assign_op = [tf.assign(weight, tf.multiply(weight, mask))
                     for weight, mask in zip(weights, masks)
                     if layer_type in weight.name]
    else:
        assign_op = [tf.assign(weight, tf.multiply(weight, mask))
                     for weight, mask in zip(weights, masks)]
    assign_all = tf.group(*assign_op)
    return assign_all


def load_prun_rate(prune_file):
    iter_rate = []
    with open(prune_file, 'r') as f:
        for line in f.readlines():
            iter_rate.append(tuple(line.split(':')))


def write_log(rate, epoch, log_dir):
    with open(os.path.join(log_dir, 'pruning_rate.txt'), 'at') as f:
        f.write('epoch:%d\t%d\t\n' % (epoch, rate))


if __name__ == '__main__':
    with tf.Session() as sess:
        model_dir = '/home/zhanghantian/models/facenet/20171128-231508'
        facenet.load_model(model_dir)
        print("load mdoel now ......")
        graph = tf.get_default_graph()
        weights = [tensor.values()[0] for tensor in graph.get_operations()
                   if tensor.name.endswith('weights')]
        cal_pruning_rate(weights)
        pruning_rate = load_prun_rate('../data/pruning_rate.txt')
        print(pruning_rate)
        for epoch, rate in pruning_rate:
            masks = get_masks(weights=weights, percentile=rate,
                              layer_type=None)
            write_log(rate, 2, os.path.expanduser('~'))
            assign_all = apply_masks(weights, masks)
            sess.run(assign_all)
            cal_pruning_rate(weights)
