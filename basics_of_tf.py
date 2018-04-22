import tensorflow as tf

with tf.device('/gpu:0'):
    x=tf.Variable(tf.random_uniform([784,784],-1,1),name="x")

sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
sess.run(tf.global_variables_initializer())
sess.run(x)
