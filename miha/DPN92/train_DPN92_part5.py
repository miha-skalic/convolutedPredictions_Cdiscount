import mxnet as mx
import math
import importlib
import sys
import numpy as np
import tensorflow as tf
import logging
from data_loader2 import queue_gen
from glob import glob

sys.path.insert(0, "./settings")

# you need logging for reports
logging.getLogger().setLevel(logging.INFO)


class tf_gen(mx.io.DataIter):
    def __init__(self, records, epoch_samples=2000, batch_size=4, augment=True):
        self.n = 0
        self.batch_size = batch_size
        self.epoch_batches = math.ceil(epoch_samples / batch_size)

        self.inpdes = mx.io.DataDesc("data", (batch_size, 3, 170, 170))
        self.outed = mx.io.DataDesc("softmax_label", (batch_size,))  # 5270 classes
        self.my_queue = queue_gen(records, epochs=1000, batch_size=batch_size, threads=12, augment=augment)

    @property
    def provide_data(self):
        return [self.inpdes]

    @property
    def provide_label(self):
        return [self.outed]

    def reset(self):
        self.n = 0

    def __next__(self):
        if self.n == self.epoch_batches:
            raise StopIteration
        else:
            self.n += 1
            xvals, yvals = next(self.my_queue)
            return mx.io.DataBatch([mx.nd.array(xvals)],
                                   [mx.nd.array(np.squeeze(yvals))])  # note that output should be 1D


num_classes = 5270
sym, arg_params, aux_params = mx.model.load_checkpoint("./trained_models/DPN92_tSGD_part4", 48)

for y in sym.get_internals():
    if y.name == "conv5_x_x__relu-sp__relu":
        break
sym = y

sym = mx.symbol.Flatten(
    mx.symbol.Pooling(data=sym, pool_type='avg', kernel=(6, 6), stride=(1, 1), pad=(0, 0), name='avg_pool'))
sym = mx.sym.Dropout(data=sym, p=0.4)
sym = mx.symbol.FullyConnected(data=sym, num_hidden=num_classes, name='logits')
sym = mx.symbol.SoftmaxOutput(data=sym, name='softmax')

devs = [mx.gpu(0)]
mod = mx.mod.Module(symbol=sym, context=devs)

batch_size = 100
train_rec = glob("/home/miha/cdisc_train_tfrecord/record_*.tfrecords")
val_rec = glob("/home/miha/cdisc_val1_tfrecord/record_*.tfrecords")

train_gen = tf_gen(records=train_rec, batch_size=batch_size, epoch_samples=2 * 10 ** 6, augment=True)
val_gen = tf_gen(records=val_rec, batch_size=batch_size, epoch_samples=10000, augment=False)

mod.fit(train_gen, val_gen,
        num_epoch=60,
        allow_missing=True,
        batch_end_callback=mx.callback.Speedometer(1, 30),
        epoch_end_callback=mx.callback.do_checkpoint("trained_models/DPN92_tSGD_170"),
        optimizer='sgd',
        arg_params=arg_params,
        aux_params=aux_params,
        optimizer_params={'learning_rate': 0.002, 'momentum': 0.9},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2.),
        begin_epoch=49,
        eval_metric="acc")  # mx.metric.TopKAccuracy(5)