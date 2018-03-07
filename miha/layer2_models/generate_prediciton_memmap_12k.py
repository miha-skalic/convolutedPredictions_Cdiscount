import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import tensorflow as tf
import pickle
import glob

from keras.optimizers import Adam
from keras.layers import Dense, GlobalMaxPooling1D, Dropout
from keras.models import Sequential
from keras.metrics import top_k_categorical_accuracy
from keras.models import load_model
from tqdm import tqdm
import pandas as pd
import numpy as np
from l2_models import get_stochastic_pool, get_stochastic_pool_probalistic
from l2_models import get_selu_maxpool

n_classes = 5270

def get_selu_maxpool(width=8000, indims=2048):
    model = Sequential()
    model.add(Dense(width, input_shape=(4, indims), activation='selu', kernel_initializer='lecun_normal'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(5270, activation='softmax', kernel_initializer='lecun_normal'))
    return model


def my_reader(filename_queue):
    reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))
    key, record_string = reader.read(filename_queue)

    features = tf.parse_single_example(record_string,
                                       features={
                                           'xception': tf.VarLenFeature(tf.float32),
                                           'xception_flip': tf.VarLenFeature(tf.float32),
                                           'inceptionresnet': tf.VarLenFeature(tf.float32),
                                           'inceptionresnet_flip': tf.VarLenFeature(tf.float32),
                                           'inceptionv3': tf.VarLenFeature(tf.float32),
                                           'inceptionv3_flip': tf.VarLenFeature(tf.float32),
                                           'idpord': tf.FixedLenFeature([1], tf.int64),
                                       })

    x_vals1 = tf.sparse_to_dense(features['xception'].indices, [8192],
                                 features['xception'].values)
    x_vals1 = tf.reshape(x_vals1, [4, 2048])

    x_vals2 = tf.sparse_to_dense(features['xception_flip'].indices, [8192],
                                 features['xception_flip'].values)
    x_vals2 = tf.reshape(x_vals2, [4, 2048])

    x_vals3 = tf.sparse_to_dense(features['inceptionresnet'].indices, [1536 * 4],
                                 features['inceptionresnet'].values)
    x_vals3 = tf.reshape(x_vals3, [4, 1536])

    x_vals4 = tf.sparse_to_dense(features['inceptionresnet_flip'].indices, [1536 * 4],
                                 features['inceptionresnet_flip'].values)
    x_vals4 = tf.reshape(x_vals4, [4, 1536])

    x_vals5 = tf.sparse_to_dense(features['inceptionv3'].indices, [2048 * 4],
                                 features['inceptionv3'].values)
    x_vals5 = tf.reshape(x_vals5, [4, 2048])

    x_vals6 = tf.sparse_to_dense(features['inceptionv3_flip'].indices, [2048 * 4],
                                 features['inceptionv3_flip'].values)
    x_vals6 = tf.reshape(x_vals6, [4, 2048])

    return x_vals1, x_vals2, x_vals3, x_vals4, x_vals5, x_vals6, features['idpord']


def tf_reader(filenames, batch_size, read_threads, num_epochs=1):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [my_reader(filename_queue)
                    for _ in range(read_threads)]

    min_after_dequeue = 0
    capacity = 5000
    example_batch1, example_batch2, example_batch3, example_batch4, \
    example_batch5, example_batch6, label_batch = tf.train.shuffle_batch_join(example_list,
                                                                              batch_size=batch_size,
                                                                              capacity=capacity,
                                                                              min_after_dequeue=min_after_dequeue,
                                                                              allow_smaller_final_batch=True)
    return example_batch1, example_batch2, example_batch3, example_batch4, example_batch5, example_batch6, label_batch


def queue_gen(read_files, epochs=100, batch_size=32, threads=6):
    xvals1, xvals2, xvals3, xvals4, xvals5, xvals6, yvals = tf_reader(read_files, batch_size=batch_size,
                                                                      num_epochs=epochs, read_threads=threads)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Initial dummy value to ensure initialization and proper model weights loading
        #
        yield None, None, None, None, None, None, None

        while True:  # TODO: write an except - shutting down the queue
            try:
                xbatch1, xbatch2, xbatch3, xbatch4, xbatch5, xbatch6, ybatch = sess.run(
                    [xvals1, xvals2, xvals3, xvals4, xvals5, xvals6, yvals])
                yield xbatch1, xbatch2, xbatch3, xbatch4, xbatch5, xbatch6, ybatch
            except:
                break

        coord.request_stop()
        coord.join(threads)

        # Finally dummy values
        while True:
            yield None, None, None, None, None, None, None


fp = np.memmap("/shared/miha/ws/competitions/p18french/ensemble/mem_maps/mihas_3model_c100_DO03_wflip.memmap",
               dtype=np.uint8, mode='w+', shape=(1768182, 5270))

y_order = pickle.load(
    open("../ensemble/idorder_layer2_xception_inceptionresnet_inceptionv3_8k12k_cp300_wflips.pkl", "rb"))
lab_to_pos = {lab: i for i, lab in enumerate(y_order)}

pbar = tqdm()

test_file = glob.glob('../tf_preprocessing/3models_cp100_wflips_test/*.tfrecords')
my_queue = queue_gen(test_file, batch_size=1000, epochs=1)

next(my_queue)
model1 = get_selu_maxpool(12000)
model1.load_weights(
    "/workspace6/mihaDL_stuff/models/xception_v0_wider12k_SELU_cp100_dropout03/xception_v0_wider12k_SELU_cp100_dropout03_800.hdf5")

model3 = get_selu_maxpool(12000, indims=1536)
model3.load_weights(
    "/workspace6/mihaDL_stuff/models/inceptionresnet_v0_wider12k_SELU_cp100_dropout03/inceptionresnet_v0_wider12k_SELU_cp100_dropout03_800.hdf5")

model5 = get_selu_maxpool(12000)
model5.load_weights(
    "/workspace6/mihaDL_stuff/models/inceptionv3_v0_wider12k_SELU_cp100_dropout03/inceptionv3_v0_wider12k_SELU_cp100_dropout03_800.hdf5")

models = [model1, model3, model5]

for x1, x2, x3, x4, x5, x6, pname in my_queue:  # xception, xception flip, inceptionresnet, inceptionresnet flip
    if x1 is None:  # We are done!
        break

    predictions1 = [models[0].predict(x) for x in (x1, x2)] + \
                   [models[1].predict(x) for x in (x3, x4)] + \
                   [models[2].predict(x) for x in (x5, x6)]

    prediction1 = np.mean(predictions1, axis=0)
    prediction1 = (prediction1 * 255.).astype(np.uint8)
    save_positions = [lab_to_pos[x] for x in pname.flatten()]

    fp[save_positions] = prediction1
    pbar.update(1)
