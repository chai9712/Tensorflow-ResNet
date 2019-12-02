from model import ResNet34
from model import AlexNet
import tensorflow as tf
from config import FLAGS
from dataset import *

def main():
    _, _, test_x, test_y, label2name = cifar_10_data(FLAGS.data_dir)
    test_x = test_x[0:200]
    test_y = test_y[0:200]
    with tf.name_scope("input_data"):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input')

    predict = AlexNet(X)
    with tf.name_scope("output_data"):
        Y = tf.placeholder(tf.int32, [None])
        Y_onehot = tf.cast(tf.one_hot(Y, 10, 1, 0), tf.float32)

    istrue = tf.equal(tf.argmax(predict, 1), tf.argmax(Y_onehot, 1))
    accuary = tf.reduce_mean(tf.cast(istrue, tf.float32))

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
        saver.restore(sess, "checkpoint/model_5.ckpt")
        var_test_acc = sess.run(accuary, feed_dict={X: test_x, Y: test_y})
        print('acc: %.5f' % var_test_acc)




if __name__ == '__main__':
    main()