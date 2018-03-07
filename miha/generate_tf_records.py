import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import pickle
import numpy as np
from tqdm import tqdm, trange
import tensorflow as tf
import multiprocessing
import math

tf_record_folder = "/home/miha/cdisc_train_tfrecord/"
os.makedirs(tf_record_folder, exist_ok=True)
tfrecords_filename = os.path.join(tf_record_folder, 'record_{:05d}.tfrecords')
train_dict = pickle.load(open("../generator_train_v1.pkl","rb"))


tcl = train_dict["classes"]
timgs = np.array(train_dict["filenames"])
my_order = np.arange(len(tcl))

np.random.seed(123)
np.random.shuffle(my_order)

tcl = tcl[my_order]
timgs = timgs[my_order]

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_and_string(img):
    img = "/home/miha/cdisc_train/" + img
    return open(img, "rb").read()  # bytearray()


shard = 0
# opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
writer = tf.python_io.TFRecordWriter(tfrecords_filename.format(shard))  # options=opts
mp = multiprocessing.Pool(10)

batch_size = 100
batches = math.ceil(len(timgs) / batch_size)

for b_indx in trange(batches):
    b_tcl = tcl[(b_indx * batch_size): ((1 + b_indx) * batch_size)]
    b_timgs = timgs[(b_indx * batch_size): ((1 + b_indx) * batch_size)]
    b_imgs = mp.map(read_and_string, b_timgs)

    for xclass, img in zip(b_tcl, b_imgs):
        example = tf.train.Example(features=tf.train.Features(feature={
            'class': _int64_feature(xclass),
            'pic_string': _bytes_feature(img)}))
        writer.write(example.SerializeToString())

    if (b_indx + 1) % 20 == 0:
        writer.close()
        shard += 1
        writer = tf.python_io.TFRecordWriter(tfrecords_filename.format(shard))  # options=opts
writer.close()