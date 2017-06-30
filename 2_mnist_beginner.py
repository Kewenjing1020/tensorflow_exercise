# mnist for beginner
# written by wenjing ke, 2017/06/27
# build a softmax regression model for mnist

# load the data sets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

import tensorflow as tf 

# build computation graph
x = tf.placeholder(tf.float32, shape = [None, 784])
y_target = tf.placeholder(tf.float32, shape = [None, 10])	# target output

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# y = wx + b
y = tf.matmul(x, w) + b
# softmax layer, get a score for each class
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits = y))

# train the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# evaluate the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for _ in range(100):
		batch = mnist.train.next_batch(100)
		train_step.run(feed_dict = {x:batch[0], y_target:batch[1]})
		print(_, accuracy.eval(feed_dict={x:mnist.test.images, y_target:mnist.test.labels}))
