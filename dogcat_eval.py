import tensorflow as tf
from datetime import datetime
import time
import math
import numpy as np

import dogcat

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_data', 'test.bin', 'Eval data')
tf.app.flags.DEFINE_string('checkpoint_dir', './train', 'Checkpoint dir')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'Eval interval secs')
tf.app.flags.DEFINE_boolean('run_once', False, 'Run once')


NUM_EXAMPLES = dogcat.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

def eval_once(saver, top_k_op, ckpt_path):
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt_path == ckpt.model_checkpoint_path:
            return ckpt.model_checkpoint_path

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(
                    sess,
                    coord=coord,
                    daemon=True,
                    start=True
                ))
            num_iter = int(math.ceil(NUM_EXAMPLES / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            precision = true_count / total_sample_count
            print('precision = %.05f : %s' % (precision, ckpt.model_checkpoint_path))
            with open('eval.log', 'a') as f:
                f.write('precision = %.05f : %s\n' % (precision, ckpt.model_checkpoint_path))


        except Exception as e:
            print(e)
            coord.request_stop(e)

    return ckpt.model_checkpoint_path


def evaluate():
    with tf.Graph().as_default() as g:
        eval_data = FLAGS.eval_data
        images, labels = dogcat.inputs(eval_data=eval_data)

        logits = dogcat.inference(images)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_average = tf.train.ExponentialMovingAverage(
            dogcat.MOVING_AVERAGE
        )
        variables_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        ckpt_path = ''
        while True:
            ckpt_path = eval_once(saver, top_k_op, ckpt_path)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):    # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    tf.app.run()
