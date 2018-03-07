import mxnet as mx
import math
import importlib
import sys
import numpy as np
import tensorflow as tf
import logging
from data_loader import queue_gen
from glob import glob

sys.path.insert(0, "./settings")

# you need logging for reports
logging.getLogger().setLevel(logging.INFO)


class tf_gen(mx.io.DataIter):
    def __init__(self, records, epoch_samples=2000, batch_size=4, augment=True):
        self.n = 0
        self.batch_size = batch_size
        self.epoch_batches = math.ceil(epoch_samples / batch_size)

        self.inpdes = mx.io.DataDesc("data", (batch_size, 3, 160, 160))
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
            # return mx.io.DataBatch([mx.nd.array(np.random.rand(batch_size, 3, 160, 160))],
            #                        [mx.nd.array(np.random.rand(batch_size, 5270))])


prefix = "./models/dpn92-5k"
network = "dpn-92"
num_classes = 5270
epoch = 0

_, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

sym = importlib.import_module('symbol_' + network).get_before_pool()
old_params = sym.list_arguments()
sym = mx.sym.Dropout(data=sym, p=0.4)
sym = mx.symbol.Flatten(
    mx.symbol.Pooling(data=sym, pool_type='avg', kernel=(5, 5), stride=(1, 1), pad=(0, 0), name='avg_pool'))
sym = mx.symbol.FullyConnected(data=sym, num_hidden=num_classes, name='logits')
sym = mx.symbol.SoftmaxOutput(data=sym, name='softmax')

devs = [mx.gpu(0)]

# # fixed_param_names set this if you want to freeze layers
mod = mx.mod.Module(symbol=sym, context=devs, fixed_param_names=old_params)
# mod = mx.mod.Module(symbol=sym, context=devs)

batch_size = 120
# train_gen = tf_gen(batch_size=batch_size, epoch_samples=2*10**6, augment=True)
train_rec = glob("/home/miha/cdisc_train_tfrecord/record_*.tfrecords")
val_rec = glob("/home/miha/cdisc_val1_tfrecord/record_*.tfrecords")

train_gen = tf_gen(records=train_rec, batch_size=batch_size, epoch_samples=2 * 10 ** 6, augment=True)
val_gen = tf_gen(records=val_rec, batch_size=batch_size, epoch_samples=10000, augment=False)

# mod.set_params(arg_params,
#                aux_params,
#                allow_missing=True)
# new_arg_params = dict({k:arg_params[k] for k in arg_params if 'logits' not in k})
# mod.bind(train_gen.provide_data, train_gen.provide_label)

mod.fit(train_gen, val_gen,
        num_epoch=3,
        allow_missing=True,
        batch_end_callback=mx.callback.Speedometer(1, 30),
        epoch_end_callback=mx.callback.do_checkpoint("trained_models/DPN92_tSGD"),
        # kvstore='device',
        optimizer='sgd',
        arg_params=arg_params,
        aux_params=aux_params,
        optimizer_params={'learning_rate': 0.01, 'momentum': 0.9},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2.),
        eval_metric="acc")  # mx.metric.TopKAccuracy(5)