"""Some function for pruning"""
import numpy as np
import tensorflow as tf
import facenet


def get_masks(weights, percentile, output_file, layer_type):
    """ Use percentile to get weight mask"""
    weights_name = [tensor.name for tensor in graph.get_operations()
                    if tensor.name.endswith('weights')]
    numpy_weights = np.array([weight.eval() for weight in weights])
    lower_thrs = [np.percentile(weight[weight != 0], percentile / 2.0)
                  if layer_type in name else np.zeros(weight.shape)
                  for weight, name in zip(numpy_weights, weights_name)]
    upper_thrs = [np.percentile(weight[weight != 0], 100 - percentile / 2.0)
                  if layer_type in name else np.zeros(weight.shape)
                  for weight, name in zip(numpy_weights, weights_name)]
    masks = [(weight <= lower_thr) + (weight >= upper_thr)
             for weight, lower_thr, upper_thr in zip(numpy_weights, lower_thrs, upper_thrs)]
    return masks


def cal_pruning_rate( weights):
    nrof_zeros = np.sum(np.array([sum(weight.eval()==0) for weight in weights]))
    total_w = np.sum(np.array([weight.eval().size for weight in weights]))
    print("The total number of weights is {} and {} zeros after pruning".format(total_w,nrof_zeros))
    print("The pruning rate is {:3f}".format(nrof_zeros * 1.0 / total_w))
    return total_w, nrof_zeros


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


if __name__ == '__main__':
    with tf.Session() as sess:
        model_dir = '/home/zhanghantian/models/pre-trained/model-20171017-182912.ckpt-80000'
        facenet.load_model(model_dir)
        print("load mdoel now ......")
        graph = tf.get_default_graph()
        weights = [tensor.values()[0] for tensor in graph.get_operations()
                    if tensor.name.endswith('weights')]
        masks = get_masks(weights=weights, percentile=50,
                          output_file='../data/mask_50', layer_type='None')
        assign_all = apply_masks(weights, masks)
        sess.run(assign_all)
        cal_pruning_rate(weights)
