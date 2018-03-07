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
import cv2
import random

train_set = pickle.load(open("../generator_train_v1.pkl", "rb"))
n_pics = {}
classes = {}
for pic, pclass in tqdm(zip(train_set["filenames"], train_set["classes"])):
    pic_name = pic.split("/")[-1].split("-")[0]
    n_pics[pic_name] = n_pics.get(pic_name, 0) + 1
    classes[pic_name] = pclass

cat_idx = {i: int(x) for i, x in enumerate(pickle.load(open("../data/class_order.pkl", "rb")))}

import multiprocessing


def cv2_read(img_string):
    return cv2.imread(img_string)


mp = multiprocessing.Pool(6)


def fetch_data():
    pic_batch = []
    pic_new = []
    pic_y = []

    pic_names = list(n_pics.keys())
    random.shuffle(pic_names)

    for c, pic_name in enumerate(pic_names):
        # product_id = d['_id']
        # category_id = d['category_id']

        pic_new += [1] + [0] * (n_pics[pic_name] - 1)
        pic_class = classes[pic_name]
        pic_y.append(pic_class)

        for pic_c in range(n_pics[pic_name]):
            pic_batch.append(
                "/workspace6/miha_misc2/train/" + "{}/{}-{}.jpg".format(cat_idx[pic_class], pic_name, pic_c))

        if (c + 1) % 100 == 0:
            pic_batch = mp.map(cv2_read, pic_batch)
            pic_batch = np.array(pic_batch)
            pic_batch = pic_batch[:, 10:170, 10:170, ::-1]  # 160 x 160 RGB
            pic_batch = ((pic_batch.astype(np.float32) / 255) - 0.5) * 2
            yield pic_batch, pic_y, pic_new
            pic_batch = []
            pic_new = []
            pic_y = []

    # final yield
    if pic_batch:
        pic_batch = mp.map(cv2_read, pic_batch)
        pic_batch = np.array(pic_batch)
        pic_batch = pic_batch[:, 10:170, 10:170, ::-1]  # 160 x 160 RGB
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
tf_record_folder = "./C_inceptionresnet_v3_cp102_train_2layer"
os.makedirs(tf_record_folder, exist_ok=True)
tfrecords_filename = os.path.join(tf_record_folder, 'record_sh{:05d}.tfrecords')

model_xception = load_model("../new_models/models/inceptionresnet_v3_part4_102-1.0037484.hdf5")
model_xception = Model(model_xception.input, model_xception.layers[-4].output)

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

    n += 100

    al_xpreds = []
    al_labels = []
    for xpred, xnp in zip(xception_pred, p_isnew):
        if xnp == 1:
            al_xpreds.append([xpred])
        elif xnp == 0:
            al_xpreds[-1].append(xpred)
        else:
            raise

    for p1, yval in zip(al_xpreds, y_batch):
        example = tf.train.Example(features=tf.train.Features(feature={
            'class': _int64_feature(yval),
            'inceptionresnet': _floats_feature(np.concatenate(p1)),
        }))
        writer.write(example.SerializeToString())

    if n == 2000:
        # Start writing to a new shard
        n = 0
        writer.close()
        shard += 1
        writer = tf.python_io.TFRecordWriter(tfrecords_filename.format(shard), options=opts)
    pbar.update(1)

writer.close()