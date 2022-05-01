"""
Original source code: https://www.tensorflow.org/hub/tutorials/biggan_generation_with_tf_hub
"""

import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm
import tensorflow_hub as hub


class BigGAN:

    def __init__(self):
        # Load a BigGAN generator module
        self.module = hub.Module('https://tfhub.dev/deepmind/biggan-deep-512/1')
        self.sess = tf.Session()
        self.inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                       for k, v in self.module.get_input_info_dict().items()}
        self.output = self.module(self.inputs)

        # Define some functions for sampling and displaying BigGAN images
        self.input_z = self.inputs['z']
        self.input_y = self.inputs['y']
        self.input_trunc = self.inputs['truncation']

        self.dim_z = self.input_z.shape.as_list()[1]
        self.vocab_size = self.input_y.shape.as_list()[1]

        # Create a TensorFlow session and initialize variables
        initializer = tf.global_variables_initializer()
        self.sess.run(initializer)

    @property
    def get_y(self):
        return self.input_y

    @property
    def get_z(self):
        return self.input_z

    @property
    def get_trunc(self):
        return self.input_trunc

    def one_hot(self, index, vocab_size):
        index = np.asarray(index)
        if len(index.shape) == 0:
            index = np.asarray([index])
        assert len(index.shape) == 1
        num = index.shape[0]
        output = np.zeros((num, vocab_size), dtype=np.float32)
        output[np.arange(num), index] = 1
        return output

    def one_hot_if_needed(self, label, vocab_size):
        label = np.asarray(label)
        if len(label.shape) <= 1:
            label = self.one_hot(label, vocab_size)
        assert len(label.shape) == 2
        return label

    def sample(self, noise, label, truncation=1., batch_size=1):
        # batch_size=8 was used by default
        noise = np.asarray(noise)
        label = np.asarray(label)
        num = noise.shape[0]
        if len(label.shape) == 0:
            label = np.asarray([label] * num)
        if label.shape[0] != num:
            raise ValueError('Got # noise samples ({}) != # label samples ({})'
                             .format(noise.shape[0], label.shape[0]))
        label = self.one_hot_if_needed(label, self.vocab_size)
        ims = []
        for batch_start in range(0, num, batch_size):
            s = slice(batch_start, min(num, batch_start + batch_size))
            feed_dict = {self.input_z: noise[s], self.input_y: label[s], self.input_trunc: truncation}
            ims.append(self.sess.run(self.output, feed_dict=feed_dict))
        ims = np.concatenate(ims, axis=0)
        assert ims.shape[0] == num
        ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
        ims = np.uint8(ims)
        return ims

    def sample_latent(self, seed, truncation, batch_size=1):
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size, self.dim_z), random_state=state)
        return truncation * values

    def truncated_z_sample(self, batch_size, truncation=1., seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size, self.dim_z), random_state=state)
        return truncation * values

    def get_latent_dims(self):
        return self.dim_z

    def partial_forward(self, z, y, truncation):
        # TODO: Ideally this should work with batch > 1. However it seems to throw an Invalid Input error.
        # seed = tf.get_default_graph().get_tensor_by_name('module/Generator_1/GenZ/G_linear/add_8:0')
        # feed_dict = {self.input_z: np.asarray(z), self.input_y: self.one_hot_if_needed(np.asarray(y), self.vocab_size), self.input_trunc: truncation}
        # return seed.eval(feed_dict=feed_dict, session=self.sess)

        seed = tf.get_default_graph().get_tensor_by_name('module_apply_default/Generator_1/GenZ/G_linear/add_8:0')
        feed_dict = {self.input_z: np.asarray(z), self.input_y: self.one_hot_if_needed(np.asarray(y), self.vocab_size), self.input_trunc: truncation}
        return seed.eval(feed_dict=feed_dict, session=self.sess)

    def write_layers(self):
        with open('biggan_layers.txt', 'w') as f:
            for item in tf.get_default_graph().get_operations():  # 147646
                f.write("%s\n" % str(item.values()))
