import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import pickle
from glob import glob

from keras.optimizers import Adam
from keras.layers import Dense, GlobalMaxPooling1D, Dropout
from keras.models import Sequential
from keras.metrics import top_k_categorical_accuracy

from keras.callbacks import ModelCheckpoint, TensorBoard

model_name = "xception_v0_wider10k_SELU_cp100_dropout03"

savepath = os.path.join("/workspace6/mihaDL_stuff/models/", model_name)
model_path = os.path.join(savepath, model_name)
os.makedirs(savepath, exist_ok=True)

n_classes = 5270


def my_reader(filename_queue, feature_names):
    reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))
    key, record_string = reader.read(filename_queue)

    features = tf.parse_single_example(record_string,
                                       features={
                                           'class': tf.FixedLenFeature([1], tf.int64),
                                           feature_names: tf.VarLenFeature(tf.float32)
                                       })
    classes = tf.sparse_to_dense([features['class']], [n_classes], [1])
    x_vals = tf.sparse_to_dense(features[feature_names].indices, [2048 * 4],
                                features[feature_names].values)
    x_vals = tf.reshape(x_vals, [4, 2048])

    return x_vals, classes


def tf_reader(filenames, feat_name, batch_size, read_threads, num_epochs=1):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [my_reader(filename_queue, feature_names=feat_name)
                    for _ in range(read_threads)]

    min_after_dequeue = 2000
    capacity = min_after_dequeue + 5 * 512
    example_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, capacity=capacity,
                                                             min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def queue_gen(read_files, feat_name, epochs=100, batch_size=32, threads=6):
    xvals, yvals = tf_reader(read_files, batch_size=batch_size, num_epochs=epochs,
                             read_threads=threads, feat_name=feat_name)

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

read_file = glob('../tf_preprocessing/C_xception_v3_cp88_train_2layer/record_*.tfrecords')
read_file_val = glob('../tf_preprocessing/C_xception_v3_cp88_val1_2layer/record*.tfrecords')

model = Sequential()
model.add(Dense(10000, input_shape=(4, 2048), activation='selu', kernel_initializer='lecun_normal'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.3))
model.add(Dense(5270, activation="softmax", kernel_initializer='lecun_normal'))
model.compile("adam", "categorical_crossentropy", metrics=[top_k_categorical_accuracy, 'accuracy'])

callbacks = [TensorBoard(log_dir='logs/{}'.format(model_name))]

my_queue = queue_gen(read_files=read_file, batch_size=512, epochs=2000, feat_name="xception")
my_val_queue = queue_gen(read_files=read_file_val, batch_size=512, epochs=2000, feat_name="xception")

model.fit_generator(generator=my_queue,
                    steps_per_epoch=300,
                    epochs=200,
                    validation_data=my_val_queue,
                    validation_steps=30,
                    callbacks=callbacks)
model.save_weights(model_path + '_{}.hdf5'.format(200))

model.compile(Adam(lr=0.0005), "categorical_crossentropy", metrics=[top_k_categorical_accuracy, 'accuracy'])
model.fit_generator(generator=my_queue,
                    steps_per_epoch=300,
                    initial_epoch=200,
                    epochs=300,
                    validation_data=my_val_queue,
                    validation_steps=30,
                    callbacks=callbacks)
model.save_weights(model_path + '_{}.hdf5'.format(300))

model.compile(Adam(lr=0.00025), "categorical_crossentropy", metrics=[top_k_categorical_accuracy, 'accuracy'])
model.fit_generator(generator=my_queue,
                    steps_per_epoch=300,
                    initial_epoch=300,
                    epochs=400,
                    validation_data=my_val_queue,
                    validation_steps=30,
                    callbacks=callbacks)
model.save_weights(model_path + '_{}.hdf5'.format(400))

model.compile(Adam(lr=0.0001), "categorical_crossentropy", metrics=[top_k_categorical_accuracy, 'accuracy'])
model.fit_generator(generator=my_queue,
                    steps_per_epoch=300,
                    initial_epoch=400,
                    epochs=500,
                    validation_data=my_val_queue,
                    validation_steps=30,
                    callbacks=callbacks)
model.save_weights(model_path + '_{}.hdf5'.format(500))

model.compile(Adam(lr=0.000075), "categorical_crossentropy", metrics=[top_k_categorical_accuracy, 'accuracy'])
model.fit_generator(generator=my_queue,
                    steps_per_epoch=300,
                    initial_epoch=500,
                    epochs=700,
                    validation_data=my_val_queue,
                    validation_steps=30,
                    callbacks=callbacks)
model.save_weights(model_path + '_{}.hdf5'.format(700))

model.compile(Adam(lr=0.00005), "categorical_crossentropy", metrics=[top_k_categorical_accuracy, 'accuracy'])
model.fit_generator(generator=my_queue,
                    steps_per_epoch=300,
                    initial_epoch=700,
                    epochs=800,
                    validation_data=my_val_queue,
                    validation_steps=30,
                    callbacks=callbacks)
model.save_weights(model_path + '_{}.hdf5'.format(800))