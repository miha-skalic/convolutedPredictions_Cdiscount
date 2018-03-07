import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import io
import bson
from skimage.data import imread
import tensorflow as tf
from keras.applications.xception import Xception

from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from PIL import Image as pil_image
from keras.utils.data_utils import GeneratorEnqueuer
import math
from keras.models import load_model
from tqdm import trange, tqdm
import numpy as np


def fetch_data():
    data = bson.decode_file_iter(open('/workspace6/miha_misc2/test.bson', 'rb'))

    pic_batch = []
    pic_new = []
    pic_y = []

    for c, d in enumerate(data):
        product_id = d['_id']

        pic_new += [1] + [0] * (len(d['imgs']) - 1)
        pic_y.append(product_id)

        for pic in d['imgs']:
            pic_batch.append(imread(io.BytesIO(pic['picture'])))

        if (c + 1) % 100 == 0:
            pic_batch = np.array(pic_batch)
            pic_batch = pic_batch[:, 10:170, 10:170]  # 160 x 160 RGB
            pic_batch = ((pic_batch.astype(np.float32) / 255) - 0.5) * 2
            yield pic_batch, pic_y, pic_new
            pic_batch = []
            pic_new = []
            pic_y = []

    # final yield
    if pic_batch:
        pic_batch = np.array(pic_batch)
        pic_batch = pic_batch[:, 10:170, 10:170]  # 160 x 160 RGB
        pic_batch = ((pic_batch.astype(np.float32) / 255.) - 0.5) * 2.
        yield pic_batch, pic_y, pic_new

    # yield Nones so we do not break keras threading
    while True:
        yield None, None, None

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Variables
tf_record_folder = "./3models_cp100_wflips_test"
os.makedirs(tf_record_folder, exist_ok=True)
tfrecords_filename = os.path.join(tf_record_folder, 'record_sh{:04d}.tfrecords')

model_xception = load_model("../new_models/models/xception_v3_part5_087-1.1903119.hdf5")
model_xception = Model(model_xception.input, model_xception.layers[-4].output)

model_inception_resnet = load_model("../new_models/models/inceptionresnet_v3_part4_102-1.0037484.hdf5")
model_inception_resnet = Model(model_inception_resnet.input, model_inception_resnet.layers[-4].output)

model_inceptionv3 = load_model("../new_models/models/inceptionv3_v3_part3_098-1.3565067.hdf5")
model_inceptionv3 = Model(model_inceptionv3.input, model_inceptionv3.layers[-4].output)

datagen = GeneratorEnqueuer(fetch_data())
datagen.start()
gen_iter = datagen.get()

# prediction loop
n = 0
shard = 0

opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
writer = tf.python_io.TFRecordWriter(tfrecords_filename.format(shard), options=opts)

pbar = tqdm()
while True:
    x_batch, y_batch, p_isnew = next(gen_iter)

    if x_batch is None:  # We iterated over all the examples
        break

    # do predictions on the batch
    xception_pred = model_xception.predict_on_batch(x_batch)
    xception_pred_flip = model_xception.predict_on_batch(x_batch[:, :, ::-1])

    inceptionresnet_pred1 = model_inception_resnet.predict_on_batch(x_batch)
    inceptionresnet_pred2 = model_inception_resnet.predict_on_batch(x_batch[:, :, ::-1])

    inceptionv3_pred1 = model_inceptionv3.predict_on_batch(x_batch)
    inceptionv3_pred2 = model_inceptionv3.predict_on_batch(x_batch[:, :, ::-1])

    n += 100

    al_xpreds1 = []
    al_xpreds2 = []
    al_incpetresnet_pred1 = []
    al_incpetresnet_pred2 = []
    al_incv3_pred1, al_incv3_pred2 = [], []
    al_labels = []

    for xpred1, xpred2, incres1, incres2, incv3a, incv3b, xnp in zip(xception_pred, xception_pred_flip,
                                                                     inceptionresnet_pred1, inceptionresnet_pred2,
                                                                     inceptionv3_pred1, inceptionv3_pred2, p_isnew):
        if xnp == 1:
            al_xpreds1.append([xpred1])
            al_xpreds2.append([xpred2])
            al_incpetresnet_pred1.append([incres1])
            al_incpetresnet_pred2.append([incres2])
            al_incv3_pred1.append([incv3a])
            al_incv3_pred2.append([incv3b])
        elif xnp == 0:
            al_xpreds1[-1].append(xpred1)
            al_xpreds2[-1].append(xpred2)
            al_incpetresnet_pred1[-1].append(incres1)
            al_incpetresnet_pred2[-1].append(incres2)
            al_incv3_pred1[-1].append(incv3a)
            al_incv3_pred2[-1].append(incv3b)
        else:
            raise

    for p1, p2, p3, p4, p5, p6, yval in zip(al_xpreds1, al_xpreds2, al_incpetresnet_pred1, al_incpetresnet_pred2,
                                            al_incv3_pred1, al_incv3_pred2, y_batch):
        example = tf.train.Example(features=tf.train.Features(feature={
            'idpord': _int64_feature(yval),  # product id
            'xception': _floats_feature(np.concatenate(p1)),
            'xception_flip': _floats_feature(np.concatenate(p2)),
            'inceptionresnet': _floats_feature(np.concatenate(p3)),
            'inceptionresnet_flip': _floats_feature(np.concatenate(p4)),
            'inceptionv3': _floats_feature(np.concatenate(p5)),
            'inceptionv3_flip': _floats_feature(np.concatenate(p6)),
        }))
        writer.write(example.SerializeToString())

    if n == 5000:
        # Start writing to a new shard
        n = 0
        writer.close()
        shard += 1
        writer = tf.python_io.TFRecordWriter(tfrecords_filename.format(shard), options=opts)
    pbar.update(1)

writer.close()
