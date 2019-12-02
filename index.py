from model import ResNet34
from model import AlexNet
import tensorflow as tf
from config import FLAGS
from dataset import *

def main():
    train_x, train_y, test_x, test_y, label2name = cifar_10_data(FLAGS.data_dir)

    #test_y = test_y.reshape(test_y.shape[0], 1)
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
        #sess.run(tf.global_variables_initializer())
        #writer = tf.summary.FileWriter("./logs", sess.graph)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
        saver.restore(sess, "checkpoint/model_5.ckpt")

        for i in range(FLAGS.train_iter):
            cur_loss = 0
            cur_acc = 0
            for images, labels in getbatch(train_x, train_y, FLAGS.batch_size):
                _, var_loss, var_pred = sess.run([optimizer, loss, predict], feed_dict={X: images, Y: labels})
                cur_loss += var_loss
                var_train_acc = sess.run(accuary, feed_dict={X: images, Y: labels})
                cur_acc += var_train_acc
            var_test_acc = sess.run(accuary, feed_dict={X: test_x, Y: test_y})

            if i % FLAGS.show_step == 0:
                print('loss: %.5f train acc: %.5f acc: %.5f' % (cur_loss/train_x.shape[0]*FLAGS.batch_size, cur_acc/train_x.shape[0]*FLAGS.batch_size, var_test_acc))

            if i % FLAGS.save_step == 0 and i != 0:
                saver.save(sess, "checkpoint/model_%d.ckpt" % i)

            # var_acc, var_pred, var_Y_onehot, var_istrue = sess.run([accuary, predict, Y_onehot, istrue],
            #                                                        feed_dict={X: test_x, Y: test_y})
            # print(var_acc, var_pred, var_Y_onehot, var_istrue)
        #writer.close()


if __name__ == '__main__':
    main()