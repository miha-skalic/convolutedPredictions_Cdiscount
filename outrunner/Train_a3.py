from keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
import cv2
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.models import Model

import pandas as pd
import os
import io
import glob

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, clear_session
from data_gen_a import Generator
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint

a3 = ['a3', 'a3_n1', 'a3_n6', 'a3_x1', 'a3_x2', 'a3_x3', 'a3_x4']
v2 = ['v2', 'v2_n1', 'v2_x1']

if os.path.isdir('checkpoints') == False:
    os.mkdir('checkpoints')

    for model in a3:
        os.mkdir(os.path.join('checkpoints', model))
        
    for model in v2:
        os.mkdir(os.path.join('checkpoints', model))
        
def Train(fz_layer = 'input_1', lr = -1, momentum = 0, epochs = 100, sub_set = 'a3'):
    
    keys = pd.read_csv('train4.csv').values
    print(len(keys))
    
    n=7000000
    train_keys = keys[:n]
    val_keys = keys[n-4:] #69896
    
    nth = -1
    val_nth = 0
    
    if sub_set == 'a3_n1':
        nth = 0
    elif sub_set == 'a3_n6':
        nth = 6
        val_nth = 1
        train_keys = train_keys[np.where(train_keys[:,2]>1)[0]]
        val_keys = val_keys[np.where(val_keys[:,2]>1)[0]]
    elif sub_set == 'a3_x1':
        train_keys = train_keys[np.where(train_keys[:,2]==1)[0]]
        val_keys   = val_keys[np.where(val_keys[:,2]==1)[0]]
    elif sub_set == 'a3_x2':
        train_keys = train_keys[np.where(train_keys[:,2]==2)[0]]
        val_keys   = val_keys[np.where(val_keys[:,2]==2)[0]]
    elif sub_set == 'a3_x3':
        train_keys = train_keys[np.where(train_keys[:,2]==3)[0]]
        val_keys   = val_keys[np.where(val_keys[:,2]==3)[0]]
    elif sub_set == 'a3_x4':
        train_keys = train_keys[np.where(train_keys[:,2]==4)[0]]
        val_keys   = val_keys[np.where(val_keys[:,2]==4)[0]]        
        
    print(train_keys.shape)
    print(val_keys.shape)

    clear_session()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list= '0'
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45
    set_session(tf.Session(config=config))

    # In[8]:
    width = 171
    up_width = 203
    input_shape=(width, width, 3)
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    x = base_model.output
    predictions = Dense(5270, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    #model.summary()
   

    epoch_r        = 0.1 # %epoch
    #fz_layer       = 'input_1' #'predictions' #'mixed_5b' #'last_conv_5270' #
    batch_size     = 70
    accum_iters    = 1
    crop           = True
    steps          = int(len(train_keys)*epoch_r)//batch_size
    val_batch_size = 100
    steps_val      = len(val_keys)//val_batch_size #
    #lr             = -1
    #momentum       = momentum

    gen    = Generator(batch_size,   train_keys, hflip_prob=0.5, color_var=0.2, 
                       crop=crop, width=width, up_width=up_width, nth=nth, rgb = True)
    genVal = Generator(val_batch_size, val_keys, hflip_prob=0, color_var=0, 
                       crop=False, width=width, up_width=up_width, train=False, nth = val_nth, rgb = True)

    for layer in model.layers:
        if layer.name == fz_layer:
            break
        layer.trainable = False

    model.compile(optimizer=SGD(lr=10**(lr),momentum=momentum, nesterov=False),loss='categorical_crossentropy',metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint('checkpoints/%s/{epoch:02d}-{loss:.3f}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}-lr%.01f-m%d-%s-s%d-%db%d.h5'%(sub_set, lr, momentum*10, fz_layer, steps, accum_iters, batch_size))

    fl = glob.glob('checkpoints/%s/*'%sub_set)
    if len(fl)>0:
        weights_file = max(fl, key=os.path.getmtime)
        model.load_weights(weights_file)
    elif sub_set != 'a3':
        fl = sorted(glob.glob('checkpoints/a3/*'), key=os.path.getmtime)
        weights_file = fl[81]
        model.load_weights(weights_file)
    else:
        weights_file = 'imagenet pretrain weight'

    print('keys         :', len(keys))
    print('train_keys   :', len(train_keys), batch_size, steps)
    print('val_keys     :', len(val_keys), val_batch_size, steps_val)
    print('weights      :', weights_file)
    print('learning rate:', 10**(lr))
    print('momentum     :', momentum)
    print('batch_size   :', batch_size)

    model.fit_generator(gen, steps,#len(train_keys)//batch_size,
                        epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint], workers=4,
                        validation_data = genVal,
                        validation_steps = steps_val)

Train(fz_layer = 'predictions', lr = -0.5, momentum = 0.9, epochs = 1,   sub_set = 'a3')
Train(fz_layer = 'input_1',     lr = -1,   momentum = 0. , epochs = 100, sub_set = 'a3')
Train(fz_layer = 'input_1',     lr = -1.5, momentum = 0. , epochs = 50,  sub_set = 'a3')
Train(fz_layer = 'input_1',     lr = -2,   momentum = 0. , epochs = 20,  sub_set = 'a3')
Train(fz_layer = 'input_1',     lr = -2.5, momentum = 0. , epochs = 10,  sub_set = 'a3')

for sub_set in a3[1:]:
    Train(fz_layer = 'input_1',     lr = -1,   momentum = 0. , epochs = 30,  sub_set = sub_set)
    Train(fz_layer = 'input_1',     lr = -1.5, momentum = 0. , epochs = 50,  sub_set = sub_set)
    Train(fz_layer = 'input_1',     lr = -2,   momentum = 0. , epochs = 20,  sub_set = sub_set)
    Train(fz_layer = 'input_1',     lr = -2.5, momentum = 0. , epochs = 10,  sub_set = sub_set)
