import tensorflow as tf
import os, shutil

def save():
        if os.path.exists("logistic_logs/"):
                shutil.rmtree("logistic_logs/")
        
        w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
        w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, 'logistic_logs/my_test_model')
 
# This will save following files in Tensorflow v >= 0.11
# my_test_model.data-00000-of-00001
# my_test_model.index
# my_test_model.meta
# checkpoint

def restore():
        with tf.Session() as sess:    
            saver = tf.train.import_meta_graph('logistic_logs/my_test_model.meta')
            saver.restore(sess,tf.train.latest_checkpoint('logistic_logs/'))
            print(sess.run('w1:0'))
            #Model has been restored. Above statement will print the saved value of w1.
            #How to access saved variable/Tensor/placeholders
            graph = tf.get_default_graph()
            w1 = graph.get_tensor_by_name("w1:0")
            print(sess.run(w1))
            ## How to access saved operation
            #op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
