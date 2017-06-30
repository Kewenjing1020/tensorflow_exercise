# created by wenjing ke, 2017/06/20
# for tensorflow beginner to get familiar with tf.variable_scope()

import tensorflow as tf 

with tf.variable_scope("foo"):
	v = tf.get_variable("v",[2])
assert v.name == "foo/v:0"

#version 1
variable_dict = {
	"conv1_w" : tf.Variable(tf.random_normal([5,5,32,32]), name = "conv1_w")
	"conv1_b" : tf.Variable(tf.zeros(32), name = "conv1_b")
	"conv2_w" : tf.Variable(tf.random_normal([5,5,32,32]), name = "conv2_w")
	"conv2_b" : tf.Variable(tf.zeros(32), name = "conv2_b")
}

def my_image_filter(input_image, variable_dict):
	conv1 = tf.nn.conv2d(input_image, variable_dict["conv1_w"], strides = [1,1,1,1], padding = "SAME")
	relu1 = tf.nn.relu(conv1 + variable_dict["conv1_b"]) 
	conv2 = tf.nn.conv2d(relu1, variable_dict["conv2_w"], strides = [1,1,1,1], padding = "SAME")
	return relu2 = tf.nn.relu(conv2+Variable["conv2_b"])

# call to my_image_filter
result1 = my_image_filter(image1, variable_dict)
result2 = my_image_filter(image2, variable_dict)


# version 2, use variable_scope
def conv_relu(input, kernel_shape, bias_shape):
	# tf.get_variable(name, shape=none, dtype=none, initializer=None,regulizer=None, trainable=True)
	# create variable named "weights"
	weights = tf.get_variable("weights", kernel_shape, initializer = random_normal_initializer())
	# create variable named 'biases'
	biases = tf.get_variable("biases", bias_shape, initializer = constant_initializer(0.0))
	conv = tf.nn.conv2d(input, weights, strides = [1,1,1,1], padding = "SAME")
	return tf.nn.relu(conv+biases)

def my_image_filter2(input_image):
	with tf.variable_scope("conv1"):
		# variable created here will be "conv1/weights", "conv1_biases"
		relu1 = conv_relu(input_image, [5,5,32,32],[32])
	with tf.variable_scope("conv2"):
		return relu2 = conv_relu(relu1, [5,5,32,32], [32])
	
# call to my_image_filter2
result3 = my_image_filter2(image1)
result4 = my_image_filter2(image2)
# raise error: conv/weigts already exists


# with tf.variable_scope("image_filter") as scope:
# 	result3 = my_image_filter2(image1)
# 	scope.reuse_variables()		 # enable share variables
# 	result4 = my_image_filter2(image2)
