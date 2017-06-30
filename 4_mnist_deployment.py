# created by wenjing ke, 2017/06/28
# deploy the mnist net on a given image
# "recognition of hand-writing letter"

import tensorflow as tf
import numpy as np
import cv2

# prepare input data
img_path = './mnist_test_image/individualImage_006.png'
img = cv2.imread(img_path, 0)
input_x = np.reshape(img, [1, 784])

with tf.Session() as sess:
	# load metagraph and checkpoint file
	saver = tf.train.import_meta_graph('./mnist_log/metaGraph/my-model.meta')
	saver.restore(sess, './mnist_log/model.cpkt-19000')
	# output of model
	y_conv = sess.graph.get_tensor_by_name("y_conv:0")
	# feed the input, run the model
	# input_x:0, keep_prob:0 are tensor_name stored in graph
	prediction = tf.argmax(y_conv, 1)
	prediction, y_conv = sess.run([prediction, y_conv], {'input_x:0': input_x, "keep_prob:0": 1.0})
	# print the scores
	print prediction, y_conv

