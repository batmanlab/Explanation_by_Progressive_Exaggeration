import tensorflow as tf
import sys 
import os
from DenseNet_Layers import *
from tensorflow.contrib.layers import flatten
import pdb

def pretrained_classifier(inputae, n_label, reuse=False,  name='classifier', isTrain = False):
    padw = 3
    n_filters = 64
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        pad_input = tf.pad(inputae, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT") 
        conv1, _conv, _bn = conv_2d_BN_Relu(pad_input, n_filters, 7, 2, 'VALID',isTrain)
        padw = 1
        pad_conv1 = tf.pad(conv1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")     
        pool1 = Max_Pooling(pad_conv1, pool_size=[3,3], stride=2)
        
        block1 =  dense_block(pool1, nb_layers=6, layer_name='dense_1', isTrain=isTrain, filters=n_filters)
        transition1 = transition_layer(block1, isTrain, scope='trans_1')
        
        block2 =  dense_block(transition1, nb_layers=12, layer_name='dense_2', isTrain=isTrain, filters=n_filters)
        transition2 = transition_layer(block2, isTrain, scope='trans_2')
        
        block3 = dense_block(transition2, nb_layers=24, layer_name='dense_3', isTrain=isTrain, filters=n_filters)
        transition3 = transition_layer(block3, isTrain, scope='trans_3')

        block4 = dense_block(transition3, nb_layers=16, layer_name='dense_final', isTrain=isTrain, filters=n_filters)

        bn = batch_norm(block4, is_training=isTrain, name='linear_batch')
        rel = tf.nn.relu(bn)
        shape = rel.get_shape().as_list()
        gap = Average_pooling(rel, pool_size=[shape[1],shape[2]], stride=1)
        flat = flatten(gap)
        pred1 = tf.layers.dense(flat, units=n_label, name='linear')
        
        if isTrain == False:
            print(isTrain)
            pred = tf.stop_gradient(pred1) 
        prediction = tf.nn.sigmoid(pred1)
        pred_y = tf.argmax(prediction, 1)
        return pred1, prediction, flat