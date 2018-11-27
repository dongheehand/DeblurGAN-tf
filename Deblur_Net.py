import tensorflow as tf
from ops import *
import numpy as np
from data_loader import dataloader
from vgg19 import Vgg19


class Deblur_Net():
    
    def __init__(self, args):
        
        self.data_loader = dataloader(args)
        
        self.channel = args.channel
        self.n_feats = args.n_feats
        self.in_memory = args.in_memory
        self.mode = args.mode
        self.batch_size = args.batch_size      
        self.num_of_down_scale = args.num_of_down_scale
        self.gen_resblocks = args.gen_resblocks
        self.discrim_blocks = args.discrim_blocks
        self.vgg_path = args.vgg_path
        
        self.learning_rate = args.learning_rate
        self.decay_step = args.decay_step
        
    def down_scaling_feature(self, name, x, n_feats):
        x = Conv(name = name + 'conv', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats * 2, strides = 2, padding = 'SAME')
        x = instance_norm(name = name + 'instance_norm', x = x, dim = n_feats * 2)
        x = tf.nn.relu(x)
        
        return x
    
    def up_scaling_feature(self, name, x, n_feats):
        x = Conv_transpose(name = name + 'deconv', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats // 2, fraction = 2, padding = 'SAME')
        x = instance_norm(name = name + 'instance_norm', x = x, dim = n_feats // 2)
        x = tf.nn.relu(x)
        
        return x
    
    def res_block(self, name, x, n_feats):
        
        _res = x
        
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode = 'REFLECT')
        x = Conv(name = name + 'conv1', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats, strides = 1, padding = 'VALID')
        x = instance_norm(name = name + 'instance_norm1', x = x, dim = n_feats)
        x = tf.nn.relu(x)
        
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode = 'REFLECT')
        x = Conv(name = name + 'conv2', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats, strides = 1, padding = 'VALID')
        x = instance_norm(name = name + 'instance_norm2', x = x, dim = n_feats)
        
        x = x + _res
        
        return x
    
    def generator(self, x, reuse = False, name = 'generator'):
        
        with tf.variable_scope(name_or_scope = name, reuse = reuse):
            _res = x
            x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], mode = 'REFLECT')
            x = Conv(name = 'conv1', x = x, filter_size = 7, in_filters = self.channel, out_filters = self.n_feats, strides = 1, padding = 'VALID')
            x = instance_norm(name = 'inst_norm1', x = x, dim = self.n_feats)
            x = tf.nn.relu(x)
            
            for i in range(self.num_of_down_scale):
                x = self.down_scaling_feature(name = 'down_%02d'%i, x = x, n_feats = self.n_feats * (i + 1))

            for i in range(self.gen_resblocks):
                x = self.res_block(name = 'res_%02d'%i, x = x, n_feats = self.n_feats * (2 ** self.num_of_down_scale))

            for i in range(self.num_of_down_scale):
                x = self.up_scaling_feature(name = 'up_%02d'%i, x = x, n_feats = self.n_feats * (2 ** (self.num_of_down_scale - i)))

            x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], mode = 'REFLECT')
            x = Conv(name = 'conv_last', x = x, filter_size = 7, in_filters = self.n_feats, out_filters = self.channel, strides = 1, padding = 'VALID')
            x = tf.nn.tanh(x)
            x = x + _res
            x = tf.clip_by_value(x, -1.0, 1.0)
            
            return x
    
    def discriminator(self, x, reuse = False, name = 'discriminator'):
        
        with tf.variable_scope(name_or_scope = name, reuse = reuse):
            x = Conv(name = 'conv1', x = x, filter_size = 4, in_filters = self.channel, out_filters = self.n_feats, strides = 2, padding = "SAME")
            x = instance_norm(name = 'inst_norm1', x = x, dim = self.n_feats)
            x = tf.nn.leaky_relu(x)
            
            prev = 1
            n = 1
            
            for i in range(self.discrim_blocks):
                prev = n
                n = min(2 ** (i+1), 8)
                x = Conv(name = 'conv%02d'%i, x = x, filter_size = 4, in_filters = self.n_feats * prev, out_filters = self.n_feats * n, strides = 2, padding = "SAME")
                x = instance_norm(name = 'instance_norm%02d'%i, x = x, dim = self.n_feats * n)
                x = tf.nn.leaky_relu(x)
                
            prev = n
            n = min(2**self.discrim_blocks, 8)
            x = Conv(name = 'conv_d1', x = x, filter_size = 4, in_filters = self.n_feats * prev, out_filters = self.n_feats * n, strides = 1, padding = "SAME")
            x = instance_norm(name = 'instance_norm_d1', x = x, dim = self.n_feats * n)
            x = tf.nn.leaky_relu(x)
            
            x = Conv(name = 'conv_d2', x = x, filter_size = 4, in_filters = self.n_feats * n, out_filters = 1, strides = 1, padding = "SAME")
            x = tf.nn.sigmoid(x)
            
            return x
    
        
    def build_graph(self):
        
        if self.in_memory:
            self.blur = tf.placeholder(name = "blur", shape = [None, None, None, self.channel], dtype = tf.float32)
            self.sharp = tf.placeholder(name = "sharp", shape = [None, None, None, self.channel], dtype = tf.float32)
            
            x = self.blur
            label = self.sharp
        
        else:
            self.data_loader.build_loader()
            
            if self.mode == 'test_only':
                x = self.data_loader.next_batch
                label = tf.placeholder(name = 'dummy', shape = [None, None, None, self.channel], dtype = tf.float32)

            elif self.mode == 'train' or self.mode == 'test':
                x = self.data_loader.next_batch[0]
                label = self.data_loader.next_batch[1]
        
        self.epoch = tf.placeholder(name = 'train_step', shape = None, dtype = tf.int32)
        
        x = (2.0 * x / 255.0) - 1.0
        label = (2.0 * label / 255.0) - 1.0
        
        self.gene_img = self.generator(x, reuse = False)
        self.real_prob = self.discriminator(label, reuse = False)
        self.fake_prob = self.discriminator(self.gene_img, reuse = True)
        
        epsilon = tf.random_uniform(shape = [self.batch_size, 1, 1, 1], minval = 0.0, maxval = 1.0)
        
        interpolated_input = epsilon * label + (1 - epsilon) * self.gene_img
        gradient = tf.gradients(self.discriminator(interpolated_input, reuse = True), [interpolated_input])[0]
        GP_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_mean(tf.square(gradient), axis = [1, 2, 3])) - 1))
        
        d_loss_real = - tf.reduce_mean(self.real_prob)
        d_loss_fake = tf.reduce_mean(self.fake_prob)
        
        if self.mode == 'train':
            self.vgg_net = Vgg19(self.vgg_path)
            self.vgg_net.build(tf.concat([label, self.gene_img], axis = 0))
            self.content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.vgg_net.relu3_3[self.batch_size:] - self.vgg_net.relu3_3[:self.batch_size]), axis = 3))

            self.D_loss = d_loss_real + d_loss_fake + 10.0 * GP_loss
            self.G_loss = - d_loss_fake + 100.0 * self.content_loss

            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            lr = tf.minimum(self.learning_rate, tf.abs(2 * self.learning_rate - (self.learning_rate * tf.cast(self.epoch, tf.float32) / self.decay_step)))
            self.D_train = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.D_loss, var_list = D_vars)
            self.G_train = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.G_loss, var_list = G_vars)
            
            logging_D_loss = tf.summary.scalar(name = 'D_loss', tensor = self.D_loss)
            logging_G_loss = tf.summary.scalar(name = 'G_loss', tensor = self.G_loss)
        
        self.PSNR = tf.reduce_mean(tf.image.psnr(((self.gene_img + 1.0) / 2.0), ((label + 1.0) / 2.0), max_val = 1.0))
        self.ssim = tf.reduce_mean(tf.image.ssim(((self.gene_img + 1.0) / 2.0), ((label + 1.0) / 2.0), max_val = 1.0))
        
        logging_PSNR = tf.summary.scalar(name = 'PSNR', tensor = self.PSNR)
        logging_ssim = tf.summary.scalar(name = 'ssim', tensor = self.ssim)
        
        self.output = (self.gene_img + 1.0) * 255.0 / 2.0
        self.output = tf.round(self.output)
        self.output = tf.cast(self.output, tf.uint8)

