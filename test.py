from model import ResNet34
import tensorflow as tf
import config as cfg
from dataset import *

def main():
    labels = [[[0, 1]], [[1, 0]]]
    pred = [[[-1, 0]], [[1, -1]]]
    y = tf.placeholder(tf.float64, [None, 1, 2])
    y_ = tf.placeholder(tf.float64, [None, 1, 2])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_var = sess.run(loss, feed_dict={y: labels, y_: pred})
        print(loss_var)


if __name__ == '__main__':
    main()