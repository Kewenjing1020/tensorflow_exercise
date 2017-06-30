# written by wenjing ke, 2017/06/26
# linear regression y = wx + b
# used for optimizing y_data = x_data + 20 * np.sin(x_data/10)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.arange(100, step = .1)
y_data = x_data + 20 * np.sin(x_data/10)

plt.plot(x_data, y_data)

batch_size = 100
n_samples = 1000
# reshape the input data, convert to tensor
x_data = np.reshape(x_data, (n_samples,1))
y_data = np.reshape(y_data, (n_samples, 1))


x = tf.placeholder(tf.float32, shape = (batch_size, 1))
y = tf.placeholder(tf.float32, shape = (batch_size, 1))

with tf.variable_scope("linear-regression"): # variable_scope add prefix for variables
	w = tf.get_variable("weight", (1, 1), initializer = tf.random_normal_initializer())	# name = "linear-regression/weight:0"
	b = tf.get_variable("bias", (1,), initializer = tf.constant_initializer(0.0))
	y_pred = tf.matmul(x,w) + b 	# y_pred = w*x + b

loss = tf.reduce_sum((y - y_pred)**2/n_samples)

# [tensorboard] add loss to summary
tf.summary.scalar('loss', loss)

# optimizer
opt_operation = tf.train.AdamOptimizer().minimize(loss)

#[tensorboard] merge all the summaries
merged = tf.summary.merge_all()

tensorboard_path = "./logs/linear_regression"

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#[tensorboard] create a filewriter for the summaries
	train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
	# train the optimizer in loop
	for i in range (5000):
		# prepare feed data
		indices = np.random.choice(n_samples, batch_size)
		x_batch, y_batch = x_data[indices], y_data[indices]
		# run optimizer
		_, loss_val, summary = sess.run([opt_operation,loss, merged], feed_dict = {x:x_batch, y:y_batch})
		if i % 100 == 0:
			# add summary to filewrite, log variable changement over time
			train_writer.add_summary(summary, i)
			print(i, loss_val)
	# get final result
	y_pred, w, b = sess.run([y_pred, w, b], feed_dict = {x:x_batch})
	print 'w = ', w, ', b = ', b
	# after release the project, tape in -> $ tensorboard --logdir = "./logs/linear_regression"
	# open your browser and go to -> http://localhost:6006/
	print 'tape in \"tensorboard --logdir=\"./logs/linear_regression\"'
	print 'open your browser and go to -> http://localhost:6006/ to take a look on tensorboard'


plt.plot(x_batch, y_pred)
plt.show()
# [tensorboard] close the filewriter
train_writer.close()


