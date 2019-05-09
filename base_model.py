from functions import *
import tensorflow.contrib.layers as layers

import config


class Basemodel(object):

    def __init__(self, mode='train'):

        self.V = config.CNN_DIM
        self.VP = config.ATTENT_DIM
        self.QP = config.ATTENT_DIM
        self.H = config.QUESTION_RNN_DIM
        self.E = config.GLOVE_DIM
        self.A = config.VQA_ANS_OUTPUT
        self.q_max = config.QUESTION_MAX_LENGTH
        self.R=196

        if mode == 'train':
            self.dropout = True
            self.trainable = True
        else:
            self.dropout = False
            self.trainable = False
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.prepare()
        self.build_vqa_module()
    def prepare(self):
        self.conv_kernel_initializer=layers.xavier_initializer()
        if config.conv_kernel_regularizer_scale>0:
            self.conv_kernel_regularizer=layers.l2_regularizer(scale=config.conv_kernel_regularizer_scale)
        else:
            self.conv_kernel_regularizer=None
        if config.conv_activity_regularizer_scale>0:
            self.conv_activity_regularizer=layers.l1_regularizer(scale=config.conv_activity_regularizer_scale)
        else:
            self.conv_activity_regularizer=None

        self.fc_kernel_initializer = tf.random_uniform_initializer(
            minval=-config.fc_kernel_initializer_scale,
            maxval=config.fc_kernel_initializer_scale)
        if self.trainable and config.fc_kernel_regularizer_scale>0:
            self.fc_kernel_regularizer = layers.l2_regularizer(
                scale=config.fc_kernel_regularizer_scale)
        else:
            self.fc_kernel_regularizer=None
        if self.trainable and config.fc_activity_regularizer_scale > 0:
            self.fc_activity_regularizer = layers.l1_regularizer(
                scale = config.fc_activity_regularizer_scale)
        else:
            self.fc_activity_regularizer = None

    def conv2d(self,
               inputs,
               filters,
               kernel_size = (3, 3),
               strides = (1, 1),
               activation = tf.nn.relu,
               use_bias = True,
               name = None):
        if activation is not None:
            activity_regularizer = self.conv_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.conv2d(
            inputs = inputs,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding='same',
            activation = activation,
            use_bias = use_bias,
            trainable = self.trainable,
            kernel_initializer = self.conv_kernel_initializer,
            kernel_regularizer = self.conv_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def max_pool2d(self,
                   inputs,
                   pool_size = (2, 2),
                   strides = (2, 2),
                   name = None):
        return tf.layers.max_pooling2d(
            inputs = inputs,
            pool_size = pool_size,
            strides = strides,
            padding='same',
            name = name)

    def dense(self,
              inputs,
              units,
              activation = tf.nn.relu,
              use_bias = True,
              name = None,
              reuse = False):
        if activation is not None:
            activity_regularizer = self.fc_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.dense(
            inputs = inputs,
            units = units,
            activation = activation,
            use_bias = use_bias,
            trainable = self.trainable,
            kernel_initializer = self.fc_kernel_initializer,
            kernel_regularizer = self.fc_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name,
            reuse = reuse)



    def batch_norm(self,
                   inputs,
                   name = None):
        return tf.layers.batch_normalization(
            inputs = inputs,
            training = self.trainable,
            trainable = self.trainable,
            name = name
        )

    def signed_sqaure_root_and_l2(self,X):
        s_X=tf.sqrt(tf.nn.relu(X))-tf.sqrt(tf.nn.relu(-X))
        s_l_X=tf.nn.l2_normalize(s_X,dim=-1)
        return s_l_X

    def mfb_vector_fusion(self, x, y,  name_scope, proj_dim=5000, output_dim=1000,
                          factor_num=5):
        with tf.variable_scope(name_scope):
            x_proj = self.dense(x,proj_dim,activation=None,name='x_proj')
            y_proj = self.dense(y, proj_dim, activation=None,name='y_proj')
        f_j = tf.multiply(x_proj, y_proj)
        if self.dropout:
            f_j = tf.nn.dropout(f_j,keep_prob=self.keep_prob)
        f_j_reshape = tf.reshape(f_j, [-1, output_dim, factor_num])
        f_j_sum = tf.reduce_sum(f_j_reshape, axis=2,keep_dims=True)
        f_j_sum=tf.reshape(f_j_sum,[-1,output_dim])
        f_j_sum = tf.nn.l2_normalize(f_j_sum,dim=-1)
        return f_j_sum

    def mfb_matrix_fusion(self, x_proj_matrix, y_proj_matrix, region_num, output_dim, factor_num):

        F_j = tf.multiply(x_proj_matrix, y_proj_matrix)
        if self.dropout:
            F_j = tf.nn.dropout(F_j, keep_prob=self.keep_prob)
        F_j_reshape = tf.reshape(F_j, [-1, region_num, output_dim, factor_num])
        F_j_sum = tf.reduce_sum(F_j_reshape, axis=3,keep_dims=True)
        F_j_sum_reshape = tf.reshape(F_j_sum, [-1, region_num, output_dim])
        F_j_sum_reshape_l2=tf.nn.l2_normalize(F_j_sum_reshape,axis=-1)
        return F_j_sum_reshape_l2