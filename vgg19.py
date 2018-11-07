import os
import tensorflow as tf

import numpy as np
import time
import inspect

'''

I modified the source code from https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
to use generally VGG19 pre-trained model

1. I removed input size limitation
2. I separated relu_layer and conv_layer
3. I removed the fully connected layer because these layer are no need for image style transfer

'''


VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [-1, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = ((rgb + 1) * 255.0) / 2.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value = rgb_scaled)
#        assert red.get_shape().as_list()[1:] == [224, 224, 1]
#        assert green.get_shape().as_list()[1:] == [224, 224, 1]
#        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
#        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.relu1_1 = self.relu_layer(self.conv1_1, "relu1_1")
        self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
        self.relu1_2 = self.relu_layer(self.conv1_2, "relu1_2")
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.relu2_1 = self.relu_layer(self.conv2_1, "relu2_1")
        self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
        self.relu2_2 = self.relu_layer(self.conv2_2, "relu2_2")
        self.pool2 = self.max_pool(self.relu2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.relu3_1 = self.relu_layer(self.conv3_1, "relu3_1")
        self.conv3_2 = self.conv_layer(self.relu3_1, "conv3_2")
        self.relu3_2 = self.relu_layer(self.conv3_2, "relu3_2")
        self.conv3_3 = self.conv_layer(self.relu3_2, "conv3_3")
        self.relu3_3 = self.relu_layer(self.conv3_3, "relu3_3")
        self.conv3_4 = self.conv_layer(self.relu3_3, "conv3_4")
        self.relu3_4 = self.relu_layer(self.conv3_4, "relu3_4")
        self.pool3 = self.max_pool(self.relu3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.relu4_1 = self.relu_layer(self.conv4_1, "relu4_1")
        self.conv4_2 = self.conv_layer(self.relu4_1, "conv4_2")
        self.relu4_2 = self.relu_layer(self.conv4_2, "relu4_2")
        self.conv4_3 = self.conv_layer(self.relu4_2, "conv4_3")
        self.relu4_3 = self.relu_layer(self.conv4_3, "relu4_3")
        self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
        self.relu4_4 = self.relu_layer(self.conv4_4, "relu4_4")
        self.pool4 = self.max_pool(self.relu4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.relu5_1 = self.relu_layer(self.conv5_1, "relu5_1")
        self.conv5_2 = self.conv_layer(self.relu5_1, "conv5_2")
        self.relu5_2 = self.relu_layer(self.conv5_2, "relu5_2")
        self.conv5_3 = self.conv_layer(self.relu5_2, "conv5_3")
        self.relu5_3 = self.relu_layer(self.conv5_3, "relu5_3")
        self.conv5_4 = self.conv_layer(self.relu5_3, "conv5_4")
        self.relu5_4 = self.relu_layer(self.conv5_4, "relu5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

#        self.fc6 = self.fc_layer(self.pool5, "fc6")
#        assert self.fc6.get_shape().as_list()[1:] == [4096]
#        self.relu6 = tf.nn.relu(self.fc6)

#        self.fc7 = self.fc_layer(self.relu6, "fc7")
#        self.relu7 = tf.nn.relu(self.fc7)

#        self.fc8 = self.fc_layer(self.relu7, "fc8")

#        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def relu_layer(self, bottom, name):
        return tf.nn.relu(bottom, name = name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
#            relu = tf.nn.relu(bias)
                   
            return bias

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

