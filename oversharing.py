import tensorflow as tf
def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_uniform_initializer(minval=-1,maxval=1)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,initializer=weight_init)
    b = tf.get_variable("b", bias_shape,initializer=bias_init)
    return tf.matmul(input, W) + b
def my_network(input):
    with tf.variable_scope("layer_1"):
        output_1 = layer(input, [784, 100], [100])
    with tf.variable_scope("layer_2"):
        output_2 = layer(output_1, [100, 50], [50])
    with tf.variable_scope("layer_3"):
        output_3 = layer(output_2, [50, 10], [10])
    return output_3
