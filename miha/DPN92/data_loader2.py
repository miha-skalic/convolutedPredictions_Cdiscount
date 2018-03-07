"""
Like data_loader but it outputs 170x170 images, additional resizing and more agressive hue
"""

import tensorflow as tf

n_classes = 5270


def my_reader(filename_queue, augment=True):
    reader = tf.TFRecordReader()
    key, record_string = reader.read(filename_queue)

    features = tf.parse_single_example(record_string,
                                       features={
                                           'class': tf.FixedLenFeature([1], tf.int64),
                                           'pic_string': tf.FixedLenFeature([], tf.string)
                                       })
    # classes = tf.reshape(tf.one_hot(features['class'], n_classes), [-1])

    img = tf.image.decode_image(features['pic_string'])
    img.set_shape([180, 180, 3])

    if augment:
        randomvar = tf.random_uniform((1,), minval=-60, maxval=+60, dtype=tf.float32)
        img = tf.contrib.image.rotate(img, randomvar)

        randomrs = tf.random_uniform((2,), minval=170, maxval=210, dtype=tf.int32)
        img = tf.image.resize_images(img, randomrs)

        img = tf.image.random_flip_left_right(tf.random_crop(img, [170, 170, 3]))
    else:
        img = img[5:175, 5:175]

    if augment:
        img = tf.image.random_hue(img, 0.1)
        img = tf.image.random_saturation(img, 0.9, 1.1)
        img = tf.image.random_brightness(img, 0.01)

    img = (tf.cast(img, tf.float32) - tf.constant([124, 117, 104], dtype=tf.float32)) * 0.0167
    img = tf.transpose(img, [2, 0, 1])
    return img, features['class']
    # return img, classes


def tf_reader(filenames, batch_size, read_threads, num_epochs=1, augment=True):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [my_reader(filename_queue, augment=augment)
                    for _ in range(read_threads)]

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 5 * 512
    example_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, capacity=capacity,
                                                             min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def queue_gen(read_files, epochs=1000, batch_size=32, threads=12, augment=True):
    xvals, yvals = tf_reader(read_files, batch_size=batch_size, num_epochs=epochs,
                             read_threads=threads, augment=augment)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:  # TODO: write an except - shutting down the queue
            xbatch, ybatch = sess.run([xvals, yvals])
            yield xbatch, ybatch

        coord.request_stop()
        coord.join(threads)