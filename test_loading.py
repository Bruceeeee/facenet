import time
import tensorflow as tf
import scipy
import numpy as np
import importlib


def main():
    with tf.Graph().as_default():
        start_time = time.time()
        with tf.Session() as sess:
            print("start loading graph in pb file ......")
            with tf.gfile.FastGFile('/home/zhanghantian/models/test_model/frozen_graph.pb', 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
            print("frozen_graph file costs {:.2f} seconds for loading".format(
                time.time() - start_time))

    with tf.Graph().as_default():
        start_time = time.time()
        with tf.Session() as sess:
            print("start loading graph in meta and ckpt ......")
            saver = tf.train.import_meta_graph(
                '/home/zhanghantian/models/test_model/pre-trained/model-20171017-182912.meta')
            saver.restore(
                sess, '/home/zhanghantian/models/test_model/pre-trained/model-20171017-182912.ckpt-80000')
            print("meta file costs {:.2f} seconds for loading".format(
                time.time() - start_time))

    # with tf.gfile.FastGFile('/home/zhanghantian/models/test_model/graph_def.pb', 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())

    # with tf.Graph().as_default():
    #     start_time = time.time()
    #     with tf.Session() as sess:
    #         print("start loading graph in pb and ckpt ......")
    #         start_time = time.time()

    #         sess.run(tf.global_variables_initializer())
    #         sess.run(tf.local_variables_initializer())
    #         saver = tf.train.Saver(tf.global_variables())
    #         saver.restore(
    #             sess, '/home/zhanghantian/models/pre-trained/model-20171017-182912.ckpt-80000')
    #         print("pb file with ckpt file loading costs {:.2f} seconds for loading".format(
    #             time.time() - start_time))

    with tf.Graph().as_default():
        print("start code defining and ckpt ")
        start_time = time.time()
        with tf.Session() as sess:
            network = importlib.import_module('models.inception_resnet_v1')
            image_batch = tf.placeholder(tf.float32, [None, 160, 160, 3])

            # Build the inference graph
            phase_train_placeholder = tf.placeholder(
                tf.bool, name='phase_train')

            prelogits, _ = network.inference(image_batch, 1,
                                             phase_train=phase_train_placeholder, bottleneck_layer_size=128,
                                             weight_decay=0.1)

            embeddings = tf.nn.l2_normalize(
                prelogits, 1, 1e-10, name='embeddings')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(
                sess, '/home/zhanghantian/models/test_model/pre-trained/model-20171017-182912.ckpt-80000')
            print("define by code, ckpt file loading cost {:.2f} seconds to load".format(
                time.time() - start_time))

    with tf.Graph().as_default():
        start_time = time.time()
        with tf.Session() as sess:
            print("start loading numpy model")
            numpy_weights = np.load('data/np_data.npy')[()]
            network = importlib.import_module('models.inception_resnet_v1')
            image_batch = tf.placeholder(tf.float32, [None, 160, 160, 3])

            # Build the inference graph
            phase_train_placeholder = tf.placeholder(
                tf.bool, name='phase_train')

            prelogits, _ = network.inference(image_batch, 1,
                                             phase_train=phase_train_placeholder, bottleneck_layer_size=128,
                                             weight_decay=0.1)

            embeddings = tf.nn.l2_normalize(
                prelogits, 1, 1e-10, name='embeddings')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            tensorLst = tf.global_variables()
            assign_all = tf.group(*[tf.assign(tensor, numpy_weights[tensor.name])
                                    for tensor in tensorLst])
            sess.run(assign_all)

            print("define by code, numpy  file loading cost {:.2f} seconds to load".format(
                time.time() - start_time))

    with tf.Graph().as_default():
        start_time = time.time()
        with tf.Session() as sess:
            print("start loading numpy sparse model")
            numpy_weights = np.load('data/np_sparse.npy')[()]
            network = importlib.import_module('models.inception_resnet_v1')
            image_batch = tf.placeholder(tf.float32, [None, 160, 160, 3])

            # Build the inference graph
            phase_train_placeholder = tf.placeholder(
                tf.bool, name='phase_train')

            prelogits, _ = network.inference(image_batch, 1,
                                             phase_train=phase_train_placeholder, bottleneck_layer_size=128,
                                             weight_decay=0.1)

            embeddings = tf.nn.l2_normalize(
                prelogits, 1, 1e-10, name='embeddings')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            tensorLst = tf.global_variables()

            assign_all = tf.group(*[tf.assign(tensor, np.reshape(np.array(numpy_weights[tensor.name].todense()), tensor.get_shape()))
                                    if 'weight' in tensor.name else tf.assign(tensor, numpy_weights[tensor.name])
                                    for tensor in tensorLst])
            sess.run(assign_all)
            print("define by code, numpy sparse  file loading cost {:.2f} seconds to load".format(
                  time.time() - start_time))


if __name__ == '__main__':
    main()
