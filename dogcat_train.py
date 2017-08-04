import tensorflow as tf
from datetime import datetime
import time
import dogcat

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train', 'Train dir')
tf.app.flags.DEFINE_integer('max_steps', 990000, 'Max steps')
tf.app.flags.DEFINE_integer('log_frequency', 10, 'Log frequency')


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.device('/cpu:0'):
            images, labels = dogcat.distorted_inputs()

        logits = dogcat.inference(images)
        loss = dogcat.loss(logits, labels)

        train_op = dogcat.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = time.time()

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = '%s: step %d, loss = %.4f (%.1f examples/sec; %.3f sec/batch)'
                    print(format_str
                          % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(gpu_options=gpu_options)
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
