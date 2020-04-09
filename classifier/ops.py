'''
Code reference: https://github.com/taki0112/Densenet-Tensorflow.git
'''
import tensorflow as tf
import tensorflow.contrib as contrib
import tensorflow.contrib.slim as slim
 
def conv2d(input_, output_dim, ks=3, s=1, stddev=0.02, padding='SAME', name='conv2d'):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None) 

def batch_norm(x, is_training, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training,scope=name)
    
def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def bottleneck_layer(x, isTrain, filters, scope):
    # print(x)
    with tf.name_scope(scope):
        x = batch_norm(x, is_training=isTrain, name=scope+'_batch1')
        #print("bn: ", x)
        x = tf.nn.relu(x)
        x = conv2d(x, filters*2, ks=1, name=scope+'_conv1')
        x = tf.layers.dropout(x, 0.5, isTrain)
        #print("conv: ", x)
        x = batch_norm(x, is_training=isTrain, name=scope+'_batch2')
        #print("bn: ", x)
        x = tf.nn.relu(x)
        x = conv2d(x, filters/2, ks=3, name=scope+'_conv2')
        x = tf.layers.dropout(x, 0.5, isTrain)
        #print("conv: ", x)
        return x

def dense_block(input_x, nb_layers, layer_name, isTrain, filters):
    with tf.name_scope(layer_name) as scope:
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, isTrain, filters, scope=layer_name + '_bottleN_' + str(0))

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = tf.concat(layers_concat, axis=3) 
            x = bottleneck_layer(x, isTrain, filters, scope=layer_name + '_bottleN_' + str(i + 1))
            layers_concat.append(x)

        x = tf.concat(layers_concat, axis=3)
        #print("concat: " , x)
        return x

def transition_layer( x, isTrain, scope):
    with tf.name_scope(scope):
        x = batch_norm(x, isTrain, name=scope+'_batch1')
        #print('TL bn: ', x)
        x = tf.nn.relu(x)
        # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')

        # https://github.com/taki0112/Densenet-Tensorflow/issues/10

        in_channel = x.get_shape().as_list()
        in_channel = in_channel[-1]* 0.5
        x = conv2d(x, in_channel, ks=1, name=scope+'_conv1')
        #print('TL conv: ', x)
        x = tf.layers.dropout(x, 0.5, isTrain)
        x = Average_pooling(x, pool_size=[2,2], stride=2)
        #print('TL Avg Pool: ', x)
        return x
        
def conv_2d_BN_Relu(inputae, n_filters, kernel, stride, padding,isTrain):
    conv = conv2d(inputae, n_filters, ks=kernel, s=stride, padding=padding)
    bn = batch_norm(conv, isTrain, name='bn')
    return tf.nn.relu(bn), conv, bn