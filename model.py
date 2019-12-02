import tensorflow as tf


def weight_variable(shape, name=None, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name, dtype=tf.float32, trainable=trainable)


def bias_variable(shape, name=None, trainable=True):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name, dtype=tf.float32, trainable=trainable)


def conv2d(input, filter, strides, padding="SAME", name=None):
    return tf.nn.conv2d(input, filter, strides, padding=padding, name=name)


def maxpool2d(input, filter, strides, padding="SAME", name=None):
    return tf.nn.max_pool(input, filter, strides, padding=padding, name=name)


def avgpool2d(input, filter, strides, padding="SAME", name=None):
    return tf.nn.avg_pool(input, filter, strides, padding=padding, name=name)


def ResNet34(input, rsizemethod = 0):
    """
    论文中的结构
    :param input: size=224,224,3
    :return:
    """
    input = tf.image.resize_images(input, [224, 224], method=rsizemethod)
    with tf.name_scope("conv1"):
        kernel = weight_variable([7, 7, 3, 64], name='kernel')
        bias = bias_variable([64], name="bias")
        conv_1 = tf.nn.relu(conv2d(input, kernel, [1, 2, 2, 1], name="conv") + bias, name="activate")
        maxpool_1 = maxpool2d(conv_1, [1, 3, 3, 1], [1, 2, 2, 1], name="pool")
    with tf.name_scope("conv2"):
        with tf.name_scope("conv2_1"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 64, 64], name="kernel")
                bias = bias_variable([64], name="bias")
                conv_2_1 = tf.nn.relu(conv2d(maxpool_1, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 64, 64], name="kernel")
                bias = bias_variable([64], name="bias")
                conv_2_1 = tf.nn.relu(conv2d(conv_2_1, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_1 = maxpool_1 + conv_2_1
        with tf.name_scope("conv2_2"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 64, 64], name="kernel")
                bias = bias_variable([64], name="bias")
                conv_2_2 = tf.nn.relu(conv2d(res_1, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 64, 64], name="kernel")
                bias = bias_variable([64], name="bias")
                conv_2_2 = tf.nn.relu(conv2d(conv_2_2, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_2 = res_1 + conv_2_2
        with tf.name_scope("conv2_3"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 64, 64], name="kernel")
                bias = bias_variable([64], name="bias")
                conv_2_3 = tf.nn.relu(conv2d(res_2, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 64, 64], name="kernel")
                bias = bias_variable([64], name="bias")
                conv_2_3 = tf.nn.relu(conv2d(conv_2_3, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_3 = res_2 + conv_2_3
    with tf.name_scope("conv3"):
        with tf.name_scope("conv3_1"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 64, 128], name="kernel")
                bias = bias_variable([128], name="bias")
                conv_3_1 = tf.nn.relu(conv2d(res_3, kernel, [1, 2, 2, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 128, 128], name="kernel")
                bias = bias_variable([128], name="bias")
                conv_3_1 = tf.nn.relu(conv2d(conv_3_1, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("shortcut"):
                kernel_line = weight_variable([3, 3, 64, 128], name="kernel_line")
                bias_line = bias_variable([128], name="bias_line")
                res_3_change = conv2d(res_3, kernel_line, [1, 2, 2, 1]) + bias_line
            with tf.name_scope("res"):
                res_4 = res_3_change + conv_3_1
        with tf.name_scope("conv3_2"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 128, 128], name="kernel")
                bias = bias_variable([128], name="bias")
                conv_3_2 = tf.nn.relu(conv2d(res_4, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 128, 128], name="kernel")
                bias = bias_variable([128], name="bias")
                conv_3_2 = tf.nn.relu(conv2d(conv_3_2, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_5 = res_4 + conv_3_2
        with tf.name_scope("conv3_3"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 128, 128], name="kernel")
                bias = bias_variable([128], name="bias")
                conv_3_3 = tf.nn.relu(conv2d(res_5, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 128, 128], name="kernel")
                bias = bias_variable([128], name="bias")
                conv_3_3 = tf.nn.relu(conv2d(conv_3_3, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_6 = res_5 + conv_3_3
        with tf.name_scope("conv3_4"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 128, 128], name="kernel")
                bias = bias_variable([128], name="bias")
                conv_3_4 = tf.nn.relu(conv2d(conv_3_3, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 128, 128], name="kernel")
                bias = bias_variable([128], name="bias")
                conv_3_4 = tf.nn.relu(conv2d(conv_3_4, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_7 = res_6 + conv_3_4
    with tf.name_scope("conv4"):
        with tf.name_scope("conv4_1"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 128, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_1 = tf.nn.relu(conv2d(res_7, kernel, [1, 2, 2, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_1 = tf.nn.relu(conv2d(conv_4_1, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("shortcut"):
                kernel_line = weight_variable([3, 3, 128, 256], name="kernel_line")
                bias_line = bias_variable([256], name="bias_line")
                res_7_change = conv2d(res_7, kernel_line, [1, 2, 2, 1]) + bias_line
            with tf.name_scope("res"):
                res_8 = res_7_change + conv_4_1
        with tf.name_scope("conv4_2"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_2 = tf.nn.relu(conv2d(res_8, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_2 = tf.nn.relu(conv2d(conv_4_2, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_9 = res_8 + conv_4_2
        with tf.name_scope("conv4_3"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_3 = tf.nn.relu(conv2d(res_9, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_3 = tf.nn.relu(conv2d(conv_4_3, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_10 = res_9 + conv_4_3
        with tf.name_scope("conv4_4"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_4 = tf.nn.relu(conv2d(res_10, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_4 = tf.nn.relu(conv2d(conv_4_4, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_11 = res_10 + conv_4_4
        with tf.name_scope("conv4_5"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_5 = tf.nn.relu(conv2d(res_11, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_5 = tf.nn.relu(conv2d(conv_4_5, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_12 = res_11 + conv_4_5
        with tf.name_scope("conv4_6"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_6 = tf.nn.relu(conv2d(res_12, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 256, 256], name="kernel")
                bias = bias_variable([256], name="bias")
                conv_4_6 = tf.nn.relu(conv2d(conv_4_6, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_13 = res_12 + conv_4_6
    with tf.name_scope("conv5"):
        with tf.name_scope("conv5_1"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 256, 512], name="kernel")
                bias = bias_variable([512], name="bias")
                conv_5_1 = tf.nn.relu(conv2d(res_13, kernel, [1, 2, 2, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 512, 512], name="kernel")
                bias = bias_variable([512], name="bias")
                conv_5_1 = tf.nn.relu(conv2d(conv_5_1, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("shortcut"):
                kernel_line = weight_variable([3, 3, 256, 512], name="kernel_line")
                bias_line = bias_variable([512], name="bias_line")
                res_13_change = conv2d(res_13, kernel_line, [1, 2, 2, 1]) + bias_line
            with tf.name_scope("res"):
                res_14 = res_13_change +conv_5_1
        with tf.name_scope("conv5_2"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 512, 512], name="kernel")
                bias = bias_variable([512], name="bias")
                conv_5_2 = tf.nn.relu(conv2d(res_14, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 512, 512], name="kernel")
                bias = bias_variable([512], name="bias")
                conv_5_2 = tf.nn.relu(conv2d(conv_5_2, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_15 = res_14 + conv_5_2
        with tf.name_scope("conv5_3"):
            with tf.name_scope("block_1"):
                kernel = weight_variable([3, 3, 512, 512], name="kernel")
                bias = bias_variable([512], name="bias")
                conv_5_3 = tf.nn.relu(conv2d(res_15, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("block_2"):
                kernel = weight_variable([3, 3, 512, 512], name="kernel")
                bias = bias_variable([512], name="bias")
                conv_5_3 = tf.nn.relu(conv2d(conv_5_3, kernel, [1, 1, 1, 1], name="conv") + bias, name="activate")
            with tf.name_scope("res"):
                res_16 = res_15 + conv_5_3
    with tf.name_scope("avg_pool"):
        avg_pool = avgpool2d(res_16, [1, 7, 7, 1], [1, 1, 1, 1], "VALID", "pool")
        line = tf.reshape(avg_pool, [-1, 512], name="line")
    with tf.name_scope("fully_connected"):
        weight = weight_variable([512, 10], name="weight")
        bias = bias_variable([10], name="bias")
        layer_34 = tf.matmul(line, weight) + bias
    # with tf.name_scope("output"):
    #     output = tf.nn.softmax(layer_34, name="softmax")
    output = layer_34
    return output


def AlexNet(input , rsizemethod = 0):
    input = tf.image.resize_images(input, [227, 227], method=rsizemethod)
    # 227 227 3
    with tf.name_scope("layer1"):
        kernel = weight_variable([11, 11, 3, 96], name="kernel")
        bias =bias_variable([96], name="bias")
        conv = tf.nn.relu(conv2d(input, kernel, [1, 4, 4, 1], padding="VALID", name="conv") + bias, name="activate")
        # 55 55 96
        pool = maxpool2d(conv, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID", name="pool")
        # 27 27 96
        lrn = tf.nn.local_response_normalization(pool, 5,
                                                 bias=2.0, alpha=0.0001, beta=0.75, name="lrn")
    with tf.name_scope("layer2"):
        kernel = weight_variable([5, 5, 96, 256], name="kernel")
        bias = bias_variable([256], name="bias")
        padding = [[0, 0], [2, 2], [2, 2], [0, 0]]
        conv = tf.nn.relu(conv2d(lrn, kernel, [1, 1, 1, 1], padding=padding, name="conv") + bias, name="activate")
        # 27 27 256
        pool = maxpool2d(conv, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID", name="pool")
        # 13 13 256
        lrn = tf.nn.local_response_normalization(pool, 5,
                                                 bias=2.0, alpha=0.0001, beta=0.75, name="lrn")
    with tf.name_scope("layer3"):
        kernel = weight_variable([3, 3, 256, 384], name="kernel")
        bias = bias_variable([384], name="bias")
        padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
        conv = tf.nn.relu(conv2d(lrn, kernel, [1, 1, 1, 1], padding=padding, name="conv") + bias, name="activate")
        # 13 13 384
    with tf.name_scope("layer4"):
        kernel = weight_variable([3, 3, 384, 384], name="kernel")
        bias = bias_variable([384], name="bias")
        padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
        conv = tf.nn.relu(conv2d(conv, kernel, [1, 1, 1, 1], padding=padding, name="conv") + bias, name="activate")
        # 13 13 384
    with tf.name_scope("layer5"):
        kernel = weight_variable([3, 3, 384, 256], name="kernel")
        bias = bias_variable([256], name="bias")
        conv = tf.nn.relu(conv2d(conv, kernel, [1, 1, 1, 1], padding=padding, name="conv") + bias, name="activate")
        # 13 13 256
        pool = maxpool2d(conv, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID", name="pool")
        # 6 6 256
    with tf.name_scope("fc1"):
        unfold = tf.reshape(pool, [-1, 9216])
        weight = weight_variable([9216, 4096], name="weight")
        bias = bias_variable([4096], name="bias")
        fc = tf.nn.relu(tf.matmul(unfold, weight) + bias, name="activate")
        drop = tf.nn.dropout(fc, rate=0.1, name="drop")
    with tf.name_scope("fc2"):
        weight = weight_variable([4096, 4096], name="weight")
        bias = bias_variable([4096], name="bias")
        fc = tf.nn.relu(tf.matmul(drop, weight) + bias, name="activate")
        drop = tf.nn.dropout(fc, rate=0.1, name="drop")
    with tf.name_scope("fc3"):
        weight = weight_variable([4096, 10], name="weight")
        bias = bias_variable([10], name="bias")
        fc = tf.nn.sigmoid(tf.matmul(drop, weight) + bias, name="activate")
    return fc


