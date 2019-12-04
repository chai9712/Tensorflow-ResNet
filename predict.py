from model import ResNet34
from model import AlexNet
import tensorflow as tf
from config import FLAGS
from dataset import *

def main():
    _, _, test_x, test_y, label2name = cifar_10_data(FLAGS.data_dir)
    test_x = test_x[0:1000]
    test_y = test_y[0:1000]
    with tf.name_scope("input_data"):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input')

    predict = AlexNet(X)
    with tf.name_scope("output_data"):
        Y = tf.placeholder(tf.int32, [None])
        Y_onehot = tf.cast(tf.one_hot(Y, 10, 1, 0), tf.float32)

    istrue = tf.equal(tf.argmax(predict, 1), tf.argmax(Y_onehot, 1))
    accuary = tf.reduce_mean(tf.cast(istrue, tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=predict))
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
        ckpt = tf.train.get_checkpoint_state("checkpoint/")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        var_test_acc, var_predict, var_loss = sess.run([accuary, predict, loss], feed_dict={X: test_x, Y: test_y})
        print('acc: %.5f' % var_test_acc)
        print(var_predict, var_loss)




if __name__ == '__main__':
    main()