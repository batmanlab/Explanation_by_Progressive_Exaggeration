import sys 
import os
sys.path.append(os.path.abspath("/pghbio/dbmi/batmanlab/singla/MICCAI_2019/GAN_Interpretability/src/common"))
import tensorflow as tf
import tensorflow.contrib as contrib
import tensorflow.contrib.slim as slim
import pdb

def spectral_normalization(name, weights, num_iters=1, update_collection=None,
                           with_sigma=False):
    """Performs Spectral Normalization on a weight tensor.
    Specifically it divides the weight tensor by its largest singular value. This
    is intended to stabilize GAN training, by making the discriminator satisfy a
    local 1-Lipschitz constraint.
    Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
    [sn-gan] https://openreview.net/pdf?id=B1QRgziT-
    Args:
    weights: The weight tensor which requires spectral normalization
    num_iters: Number of SN iterations.
    update_collection: The update collection for assigning persisted variable u.
                       If None, the function will update u during the forward
                       pass. Else if the update_collection equals 'NO_OPS', the
                       function will not update the u during the forward. This
                       is useful for the discriminator, since it does not update
                       u in the second pass.
                       Else, it will put the assignment in a collection
                       defined by the user. Then the user need to run the
                       assignment explicitly.
    with_sigma: For debugging purpose. If True, the fuction returns
                the estimated singular value for the weight tensor.
    Returns:
    w_bar: The normalized weight tensor
    sigma: The estimated singular value for the weight tensor.
    """
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
    u = tf.get_variable(name + 'u', [1, w_shape[-1]],
                      initializer=tf.truncated_normal_initializer(),
                      trainable=False)
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def _l2normalize(v, eps=1e-12):
    """l2 normize the input vector."""
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def conditional_batchnorm(x, train_phase, scope_bn, y=None, nums_class=10):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        if y == None:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        else:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[nums_class, x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[nums_class, x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
            beta, gamma = tf.nn.embedding_lookup(beta, y), tf.nn.embedding_lookup(gamma, y)
            beta = tf.reshape(beta, [-1, 1, 1, x.shape[-1]])
            gamma = tf.reshape(gamma, [-1, 1, 1, x.shape[-1]])
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments', keep_dims=True)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv(name, inputs, nums_out, k_size, strides, update_collection=None, is_sn=False):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            W = spectral_normalization("sn", W, update_collection=update_collection)
        con = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], "SAME")
    return tf.nn.bias_add(con, b)

def dense(name, inputs, nums_out, update_collection=None, is_sn=False):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            W = spectral_normalization("sn", W, update_collection=update_collection)
    return tf.nn.bias_add(tf.matmul(inputs, W), b)

def Inner_product(global_pooled, y, nums_class, update_collection=None):
    W = global_pooled.shape[-1]
    V = tf.get_variable("V", [nums_class, W], initializer=tf.glorot_uniform_initializer())
    V = tf.transpose(V)
    V = spectral_normalization("embed", V, update_collection=update_collection)
    V = tf.transpose(V)
    temp = tf.nn.embedding_lookup(V, y)
    temp = tf.reduce_sum(temp * global_pooled, axis=1, keep_dims=True)
    return temp

def G_Resblock(name, inputs, nums_out, is_training, y, nums_class, update_collection=None, is_sn = False):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = conditional_batchnorm(inputs, is_training, "bn1", y, nums_class)
        inputs = tf.nn.relu(inputs)
        H = inputs.shape[1]
        W = inputs.shape[2]
        inputs = tf.image.resize_nearest_neighbor(inputs, [H * 2, W * 2]) #up-sampling
        #print(name, ' upsample ', inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
        inputs = conditional_batchnorm(inputs, is_training, "bn2", y, nums_class)
        inputs = tf.nn.relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
        #Identity mapping
        H = temp.shape[1]
        W = temp.shape[2]
        temp = tf.image.resize_nearest_neighbor(temp, [H * 2, W * 2])#upsampling before conv in G
        temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=is_sn)
    return inputs + temp

def G_Resblock_Encoder(name, inputs, nums_out, is_training, y, nums_class, update_collection=None, is_sn = False):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = conditional_batchnorm(inputs, is_training, "bn1", y, nums_class)
        inputs = tf.nn.relu(inputs)
        inputs =  tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
        inputs = conditional_batchnorm(inputs, is_training, "bn2", y, nums_class)
        inputs = tf.nn.relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
        #Identity mapping
        temp =  tf.nn.avg_pool(temp, [1, 2, 2, 1], [1, 2, 2, 1], "SAME") #downsampling before conv in G
        temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=is_sn)
    return inputs + temp
    
def D_Resblock(name, inputs, nums_out, update_collection=None, is_down=True, is_sn = True):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = tf.nn.relu(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
        inputs = tf.nn.relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection, is_sn=is_sn)
        if is_down:
            inputs = tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME") #downsampling after 2nd conv in D
            #Identity mapping
            temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=is_sn) #replacing identity mapping with 1x1 conv
            temp = tf.nn.avg_pool(temp, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        # else:
        #     temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=True)
    return inputs + temp

def D_FirstResblock(name, inputs, nums_out, update_collection, is_down=True, is_sn = True):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection=update_collection, is_sn=is_sn)
        inputs = tf.nn.relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection=update_collection, is_sn=is_sn)
        if is_down:
            inputs =  tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            #Identity mapping
            temp =  tf.nn.avg_pool(temp, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            temp = conv("identity", temp, nums_out, 1, 1, update_collection=update_collection, is_sn=is_sn)
    return inputs + temp
