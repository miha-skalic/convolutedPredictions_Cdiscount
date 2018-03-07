from keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
import cv2
from keras.layers import Dense
from keras.models import Model

import pandas as pd
import os
from tqdm import tqdm_notebook
from skimage.data import imread
import io
import glob

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, clear_session
from data_gen import Generator
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from accumulate import Adam_accumulate, SGD_accumulate

a3 = ['a3', 'a3_n1', 'a3_n6', 'a3_x1', 'a3_x2', 'a3_x3', 'a3_x4']
v2 = ['v2', 'v2_n1', 'v2_x1']

if os.path.isdir('checkpoints') == False:
    os.mkdir('checkpoints')

    for model in a3:
        os.mkdir(os.path.join('checkpoints', model))
        
    for model in v2:
        os.mkdir(os.path.join('checkpoints', model))

def Train(fz_layer = 'input_1', lr = -1, momentum = 0, epochs = 100, sub_set = 'v2'):
 
    if sub_set == 'v2_x1':
        df2 = pd.read_csv('train_imgs.csv')
        uid = df2[df2.n_images==1]._id.values

        df = pd.read_csv('train2.csv')
        df.columns = ['sn', 'key','_id','c_id','y']
        df_train = df[:12000000]
        df_val = df[12000000:]

        train_keys = df_train[df_train._id.isin(uid)][['key', 'y']].values
        val_keys = df_val[df_val._id.isin(uid)][['key', 'y']].values
    else:
        keys = pd.read_csv('train2.csv').values[:,[1,4]]

        train_keys = keys[:12000000,:]
        val_keys = keys[12000000:,:] 

        if sub_set == 'v2_n1':
            temp = []
            for key in train_keys:
                if int(key[0][-5]) == 0:
                    temp.append(key)
            train_keys = np.array(temp)

            temp = []
            for key in val_keys:
                if int(key[0][-5]) == 0:
                    temp.append(key)
            val_keys = np.array(temp)

    clear_session()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list= '0'
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45
    set_session(tf.Session(config=config))

    width = 171
    up_width = 203

    input_shape=(width, width, 3)
    base_model = InceptionResNetV2(include_top=False, weights=None, input_shape=input_shape, pooling='avg')
    x = base_model.output
    predictions = Dense(5270, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # In[9]:
    epoch_r        = 0.1 # %epoch
    #fz_layer       = 'input_1' #'mixed_6a' #'dense_1' #
    batch_size     = 96
    accum_iters    = 1
    crop           = True
    steps          = len(train_keys)*epoch_r//batch_size
    val_batch_size = 100
    steps_val      = 100000//val_batch_size #
    #lr             = -2.5
    #momentum       = 0.

    for layer in model.layers:
        if layer.name == fz_layer :
            break
        layer.trainable = False

    #model.summary()

    gen    = Generator(batch_size,   train_keys, hflip_prob=0.5, crop=crop, width=width, up_width=up_width)
    genVal = Generator(val_batch_size, val_keys, hflip_prob=0, crop=False, width=width, up_width=up_width, train=False)

    if accum_iters > 1:
        model.compile(optimizer=SGD_accumulate(lr=10**(lr),
                      momentum=momentum, accum_iters=accum_iters),
                      loss='categorical_crossentropy',metrics=['accuracy'])
    else:
        model.compile(optimizer=SGD(lr=10**(lr),
                      momentum=momentum, nesterov=False),
                      loss='categorical_crossentropy',metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint('checkpoints/%s/{epoch:02d}-{loss:.3f}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}-lr%.01f-m%d-%s-s%d-%db%d.h5'%(sub_set, lr, momentum*10, fz_layer, steps, accum_iters, batch_size))

    # In[11]:
    #weights_file = max(glob.glob('checkpoints/InceptionResNetV2_180/*'), key=os.path.getmtime)
    #weights_file = 'checkpoints/InceptionResNetV2_215/lr-2.0_01-1.172-0.725.h5'
    #model.load_weights(weights_file, by_name=True)

    fl = glob.glob('checkpoints/%s/*'%sub_set)
    if len(fl)>0:
        weights_file = max(fl, key=os.path.getmtime)
        model.load_weights(weights_file)
    elif sub_set != 'v2':
        fl = sorted(glob.glob('checkpoints/v2/*'), key=os.path.getmtime)
        weights_file = fl[81]
        model.load_weights(weights_file)
    else:
        weights_file = 'imagenet pretrain weight'

    print('keys         :', len(keys))
    print('train_keys   :', len(train_keys))
    print('val_keys     :', len(val_keys), steps_val*val_batch_size)
    print('weights      :', weights_file)
    print('learning rate:', 10**(lr))
    print('momentum     :', momentum)
    print('batch_size   :', batch_size)

    # In[ ]:

    model.fit_generator(gen, steps, epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint], workers=4,
                        validation_data = genVal, validation_steps = steps_val)

Train(fz_layer = 'predictions', lr = -0.5, momentum = 0.9, epochs = 1,   sub_set = 'v2')
Train(fz_layer = 'input_1',     lr = -1,   momentum = 0. , epochs = 100, sub_set = 'v2')
Train(fz_layer = 'input_1',     lr = -1.5, momentum = 0. , epochs = 50,  sub_set = 'v2')
Train(fz_layer = 'input_1',     lr = -2,   momentum = 0. , epochs = 20,  sub_set = 'v2')
Train(fz_layer = 'input_1',     lr = -2.5, momentum = 0. , epochs = 10,  sub_set = 'v2')

for sub_set in v2[1:]:
    Train(fz_layer = 'input_1',     lr = -1,   momentum = 0. , epochs = 30,  sub_set = sub_set)
    Train(fz_layer = 'input_1',     lr = -1.5, momentum = 0. , epochs = 50,  sub_set = sub_set)
    Train(fz_layer = 'input_1',     lr = -2,   momentum = 0. , epochs = 20,  sub_set = sub_set)
    Train(fz_layer = 'input_1',     lr = -2.5, momentum = 0. , epochs = 10,  sub_set = sub_set)