from model import ResNet34
from model import AlexNet
from model import testNet
import tensorflow as tf
from config import FLAGS
from dataset import *
from time import time

def main():
    train_x, train_y, test_x, test_y, label2name = cifar_10_data(FLAGS.data_dir)

    test_x = test_x[0:200]
    test_y = test_y[0:200]

    with tf.name_scope("input_data"):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input')
    # predict = ResNet34(X)
    predict = AlexNet(X)
    with tf.name_scope("output_data"):
        Y = tf.placeholder(tf.int32, [None])
        Y_onehot = tf.cast(tf.one_hot(Y, 10, 1, 0), tf.float32)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=predict))
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    istrue = tf.equal(tf.argmax(predict, 1), tf.argmax(Y_onehot, 1))
    accuary = tf.reduce_mean(tf.cast(istrue, tf.float32))

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        ckpt = tf.train.get_checkpoint_state("checkpoint/")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        starttime = time()
        for ep in range(0, FLAGS.train_iter):
            cur_loss = 0
            cur_acc = 0
            for images, labels in getbatch(train_x, train_y, FLAGS.batch_size):
                var_loss, var_pred = sess.run([loss, predict], feed_dict={X: images, Y: labels})
                cur_loss += var_loss
                var_train_acc = sess.run(accuary, feed_dict={X: images, Y: labels})
                cur_acc += var_train_acc
            var_test_acc = sess.run(accuary, feed_dict={X: test_x, Y: test_y})

            if ep % FLAGS.show_step == 0:
                print(time() - starttime)
                starttime = time()
                print('loss: %.5f train acc: %.5f acc: %.5f' % (cur_loss/train_x.shape[0]*FLAGS.batch_size, cur_acc/train_x.shape[0]*FLAGS.batch_size, var_test_acc))


if __name__ == '__main__':
    main()