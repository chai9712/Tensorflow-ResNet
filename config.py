import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', r'D:\cifar\cifar-10-python\cifar-10-batches-py', '')
flags.DEFINE_integer('train_iter', 100, '')
flags.DEFINE_float('learning_rate', 0.0001, '')
flags.DEFINE_integer('batch_size', 50, '')
flags.DEFINE_integer('save_step', 1, '')
flags.DEFINE_integer('show_step', 1, '')
