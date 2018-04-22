import tensorflow as tf

X = tf.palceholder(tf.float32, [None,28,28,1])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()

#model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X,[-1,784]),W)+b)
#placeholder for correct answer
Y_ = tf.placeholder(tf.float32,[None,10])

#loss function
cross_entropy = -tf.reduce_sum(Y_*tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003) #learning rate
train_step = optimizer.minimize(cross_entropy) #loss function

sess = tf.Session()
sess.run(init)

for i in range(1000):
	#load batch of images and correct answers
	batch_X, batch_Y = mnist.train.next_batch(100)
	train_data = {X: batch_X, Y_:batch_Y}
	
	#train
	sess.run(train_step, feed_dict=train_data)
	
	#