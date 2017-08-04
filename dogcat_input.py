import tensorflow as tf
import os

IMG_ORIGINAL_SIZE = 128
IMG_SIZE = 100

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 24000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000
NUM_DATA_FILES = 1


def read_bin(filename_queue):
    class BinRecord(object):
        pass
    result = BinRecord()

    label_bytes = 1
    result.height = IMG_ORIGINAL_SIZE
    result.width = IMG_ORIGINAL_SIZE
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes])
        , tf.int32
    )

    result.uint8image = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.width, result.height, result.depth]
    )

    return result


def _gen_img_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 1

    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples * 1 * batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples * 1 * batch_size,
        )

        tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, '%02d.bin' % i)
                 for i in range(NUM_DATA_FILES)]

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_bin(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    distorted_image = tf.random_crop(reshaped_image, [IMG_SIZE, IMG_SIZE, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)
    float_image.set_shape([IMG_SIZE, IMG_SIZE, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.01
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    print('Filling queue with %d images' % min_queue_examples)

    return _gen_img_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, '%02d.bin' % i)
                     for i in range(NUM_DATA_FILES)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, eval_data)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_bin(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, IMG_SIZE, IMG_SIZE)

    float_image = tf.image.per_image_standardization(resized_image)
    float_image.set_shape([IMG_SIZE, IMG_SIZE, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.005
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return _gen_img_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)
