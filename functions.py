import tensorflow as tf 
import config

def signed_sequre_root_l2(x):
    x = tf.sqrt(x)
    x = tf.sign(x)
    x = tf.nn.l2_normalize(x, 0)
    return x

def locak_word_embedding(word_embs, inputs):
    x = tf.nn.embedding_lookup(word_embs, inputs, name='word_vector')
    return x


def weight_variable(shape,dtype=tf.float32,name=None,lamda=config.WEIGHT_DECAY):
    var=tf.get_variable(name,shape,dtype=dtype,initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamda)(var))
    return var

def bias_variable(shape,dtype=tf.float32,name=None):
	return tf.get_variable(name,shape,dtype=dtype,initializer=tf.constant_initializer(0.0))