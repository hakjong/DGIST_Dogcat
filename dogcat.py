import tensorflow as tf
import re

import dogcat_input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', '32', 'Batch size')
tf.app.flags.DEFINE_string('data_dir', './bin', 'Data dir')

IMG_SIZE = dogcat_input.IMG_SIZE
NUM_CLASSES = dogcat_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = dogcat_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = dogcat_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE = 0.9999
LEARNING_RATE = 0.00005


def _activation_summary(x):
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + 'sparsity', tf.nn.zero_fraction(x))


def _var_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _var_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16
    var = _var_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    )
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    data_dir = FLAGS.data_dir
    images, labels = dogcat_input.distorted_inputs(
        data_dir=data_dir,
        batch_size=FLAGS.batch_size
    )
    return images, labels


def inputs(eval_data):
    data_dir = FLAGS.data_dir
    images, labels = dogcat_input.inputs(
        eval_data=eval_data,
        data_dir=data_dir,
        batch_size=FLAGS.batch_size
    )
    return images, labels


def inference(images):
    with tf.variable_scope('01_conv32') as scope:
        net = tf.layers.conv2d(images, 32, [4, 4], padding='SAME', activation=tf.nn.relu, name=scope.name + '1')
        _activation_summary(net)
        net = tf.layers.conv2d(net, 32, [4, 4], padding='SAME', activation=tf.nn.relu, name=scope.name + '2')
        _activation_summary(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], name=scope.name + '3')
        _activation_summary(net)

    with tf.variable_scope('02_conv64') as scope:
        net = tf.layers.conv2d(net, 64, [4, 4], padding='SAME', activation=tf.nn.relu, name=scope.name + '1')
        _activation_summary(net)
        net = tf.layers.conv2d(net, 64, [4, 4], padding='SAME', activation=tf.nn.relu, name=scope.name + '2')
        _activation_summary(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], name=scope.name + '3')
        _activation_summary(net)

    with tf.variable_scope('03_conv128') as scope:
        net = tf.layers.conv2d(net, 128, [4, 4], padding='SAME', activation=tf.nn.relu, name=scope.name + '1')
        _activation_summary(net)
        net = tf.layers.conv2d(net, 128, [4, 4], padding='SAME', activation=tf.nn.relu, name=scope.name + '2')
        _activation_summary(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], name=scope.name + '3')
        _activation_summary(net)

    with tf.variable_scope('04_conv256') as scope:
        net = tf.layers.conv2d(net, 256, [4, 4], padding='SAME', activation=tf.nn.relu, name=scope.name + '1')
        _activation_summary(net)
        net = tf.layers.conv2d(net, 256, [4, 4], padding='SAME', activation=tf.nn.relu, name=scope.name + '2')
        _activation_summary(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], name=scope.name + '3')
        _activation_summary(net)

    with tf.variable_scope('05_local1') as scope:
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 256, activation=tf.nn.relu, name=scope.name)
        _activation_summary(net)
        net = tf.layers.dropout(net, 0.5)

    with tf.variable_scope('06_local1') as scope:
        net = tf.layers.dense(net, 256, activation=tf.nn.relu, name=scope.name)
        _activation_summary(net)
        net = tf.layers.dropout(net, 0.5)

    with tf.variable_scope('07_local1') as scope:
        model = tf.layers.dense(net, 2, name=scope.name)
        _activation_summary(net)

    model = tf.nn.softmax(model)
    return model


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size

    lr = LEARNING_RATE

    loss_average_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_average_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE,
        global_step
    )
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op