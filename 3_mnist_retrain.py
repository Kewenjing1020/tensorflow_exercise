# created by wenjing ke, 2017/06/27
# merge the code of /tensorflow/examples/tutorials/mnist/mnist_deep.py
# add the function of: 1. save and load the model
#                      2. export metagraph
#                      3. add the part of fine-tuning
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
fine_tune = False

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def deepnn(x):
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name = "y_conv")
  return y_conv, keep_prob, h_fc1_drop


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # input placeholder
  x = tf.placeholder(tf.float32, [None, 784], name = "input_x")
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob, h_fc1_drop = deepnn(x)

  # loss function
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  # optimization 
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  
  # evaluation
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  # Create a saver.
  saver = tf.train.Saver(tf.global_variables())

  # export a metagraph
  meta_graph_def = tf.train.export_meta_graph('./mnist_log/metaGraph/my-model.meta', as_text = True)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      # Save the model checkpoint periodically.
      if i % 1000 == 0:
        saver.save(sess, "./mnist_log/model.cpkt", global_step = i)

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

  if fine_tune == True: 
    # omit last layer "y_conv"
    # set new layer "y_conv_new" after h_fc1_drop
    W_fc_new = weight_variable([1024, 10])
    b_fc_new = bias_variable([10])
    y_conv_new = tf.matmul(h_fc1_drop, W_fc_new) + b_fc_new

    # optimization and evaluation
    cross_entropy_new = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv_new))
    train_step_new = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_new)
    correct_prediction_new = tf.equal(tf.argmax(y_conv_new, 1), tf.argmax(y_, 1))
    accuracy_new = tf.reduce_mean(tf.cast(correct_prediction_new, tf.float32))

    # retrain(same as the original part)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, "./mnist_log/model.cpkt-19000")
      print('test accuracy %g' % accuracy_new.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

      for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 100 == 0:
          train_accuracy_new = accuracy_new.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
          print('step %d, training accuracy %g' % (i, train_accuracy_new))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
