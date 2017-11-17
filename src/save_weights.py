import time
import importlib

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def save_weights_np(model_dir):
    start_time = time.time()
    with tf.Session() as sess:
        network = importlib.import_module('models.inception_resnet_v1')
        image_batch = tf.placeholder(tf.float32, [None, 160, 160, 3])

        # Build the inference graph
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        prelogits, _ = network.inference(image_batch, 1,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=128,
                                         weight_decay=0.1)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_dir)
        print("cost {:.2f} seconds to load".format(time.time() - start_time))
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        data = {weight.name: weight.eval() for weight in weights}
        numpy_data = np.array(data)
        np.save('../data/np_data', numpy_data)



def main():
    save_weights_np(
        model_dir='/Users/finally/Desktop/code_examples/myfacenet/models/facenet_v2/model-20171017-182912.ckpt-80000')

if __name__ == '__main__':
    main()
