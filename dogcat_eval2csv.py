import tensorflow as tf
from datetime import datetime
import time
import math
import numpy as np

import dogcat

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', './eval', 'Eval dir')
tf.app.flags.DEFINE_string('eval_data', 'tocsv.bin', 'Eval data')
tf.app.flags.DEFINE_string('checkpoint_dir', './train', 'Checkpoint dir')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, 'Eval interval secs')
tf.app.flags.DEFINE_boolean('run_once', True, 'Run once')


NUM_EXAMPLES = 12550

def eval_once(saver, summary_writer, top_k_op, summary_op):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print(ckpt.model_checkpoint_path, end=' ')
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
            with open('result.csv', 'w') as f:
                f.write('filename,label\n')
                id = 1
                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    for a in predictions[0]:
                        f.write('%d.jpg,%d\n' % (id - 1, 1 if a else 0))
                        if id == NUM_EXAMPLES:
                            break
                        id += 1
                    true_count += np.sum(predictions)
                    step += 1

            precision = true_count / NUM_EXAMPLES
            print('Done')
            #print('%s: precision @ 1 = %.05f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:
            print(e)
            coord.request_stop(e)


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

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):    # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
