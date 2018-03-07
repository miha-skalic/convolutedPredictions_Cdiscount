import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'  #'0,1,2,3'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from dataset.cdiscount_image_dataset import *
from dataset.cdiscount_feature_set_dataset import *

#import net
#from net.model.cdiscount.se_resnet50 import image_to_tensor_transform, SEResNet50 as Net
#from net.model.cdiscount.se_resnext_101_32x4d import image_to_tensor_transform, SEResnext101 as Net; DIM=2048

#from net.model.cdiscount.xception  import image_to_tensor_transform, Xception as Net; DIM=2048
#from net.model.cdiscount.resnet101  import image_to_tensor_transform, ResNet101 as Net; DIM=2048
from net.model.cdiscount.dualpathnet92  import image_to_tensor_transform, DPNet92 as Net; DIM=2688
#from net.model.cdiscount.se_resnext_101_32x4d  import image_to_tensor_transform, SEResnext101 as Net; DIM=2048

#import augmentation
#default_augment = lambda image,label,index : (image_to_tensor_transform(image),label,index)


def extract_augment(image,label,index):
    #image  = cv2.flip(image,1)
    tensor = image_to_tensor_transform(image)
    return tensor,label,index


def run_feature_extract():

    #out_dir = '/root/share/project/kaggle/cdiscount/data/feature/xcpetion-20b-flip'
    #checkpoint = '/root/share/project/kaggle/cdiscount/results/xcpetion-20b/checkpoint/00415000_model.pth'

    out_dir    = '/root/share/project/kaggle/cdiscount/data/feature/dpn92-180-05a'
    checkpoint = '/root/share/project/kaggle/cdiscount/results/dpn92-180-05a/checkpoint/00032000_model.pth'

    #out_dir    = '/root/share/project/kaggle/cdiscount/data/feature/se-resnext101-51c-flip'
    #checkpoint = '/root/share/project/kaggle/cdiscount/results/se-resnext101-51c/checkpoint/00208000_model.pth'


    ## net ---------------------------------

    net = Net(in_shape = (3,CDISCOUNT_HEIGHT,CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES, type = 'last_feature')
    net.load_state_dict(torch.load(checkpoint))


    net.cuda()
    net.eval()


    ## dataset  ----------------------------
    dataset = CDiscountImageDataset(
                                #'debug_test_id_100', 'test', mode='test',
                                #'debug_train_id_v0_5000', 'train', mode='train',

                                'test_id_1768182', 'test', mode='test',
                                #'train_id_v0_7019896', 'train', mode='train',
                                #'valid_id_v0_50000', 'train', mode='train',
                                #'train_id_v1_7019896', 'train', mode='train',
                                #'valid_id_v1_50000', 'train', mode='train',


                                transform = extract_augment )
    loader  = DataLoader(
                        dataset,
                        sampler     = SequentialSampler(dataset), #FixedSampler(test_dataset,[0,1,2,]),  #
                        batch_size  = 128*NUM_CUDA_DEVICES,  #784,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)
    num_images = len(loader.dataset)



    ## prediction  ----------------------------
    feature_dir = out_dir +'/%s'%(dataset.split)
    os.makedirs(feature_dir, exist_ok=True)

    memmap_file = feature_dir+'/features_%dx%d.uint8.memmap'%(num_images,DIM)
    label_file  = feature_dir+'/labels_%dx%d.int16.npy'%(num_images,DIM)
    txt_file    = feature_dir+'/header_%dx%d.txt'%(num_images,DIM)

    with open(txt_file,'w') as f:
        f.write('identifier    = %s\n' % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        f.write('memmap_file   = %s\n' % memmap_file)
        f.write('dataset.split = %s\n' % dataset.split)
        f.write('checkpoint    = %s\n' % checkpoint)
        f.write('\n')

    if dataset.labels is not None : np.save(label_file, np.array(dataset.labels,np.int16))
    f = np.memmap(memmap_file, dtype='uint8', mode='w+', shape=(num_images,DIM))


    n = 0
    start = timer()
    for i, (tensors, labels, indices) in enumerate(loader, 0):
        print('\r%5d:  %10d/%d (%0.0f %%)  %0.2f min'%(i, n, num_images, 100*n/num_images,
                         (timer() - start) / 60), end='',flush=True)

        # forward
        tensors  = Variable(tensors,volatile=True).cuda(async=True)
        features = data_parallel(net, tensors)
        #features = F.softmax(features,dim=1) ###

        batch_size = len(indices)
        features = features.data.cpu().numpy()
        f[n:n+batch_size] = encode_features(features)
        n += batch_size

    assert(n == len(loader.sampler) and n == num_images) #check
    print('\r%5d:  %10d/%d (%0.0f %%)  %0.2f min'%(i, n, num_images, 100*n/num_images,
                         (timer() - start) / 60), end='\n',flush=True)



    if 1 : ## for internal check (please ignore this) ##-----------------------------
        print('')
        product_scores =[]
        product_scores.append( f[0:3].mean(0)   )
        product_scores.append( f[3]             )
        product_scores.append( f[4]             )
        product_scores.append( f[5]             )
        product_scores.append( f[6]             )
        product_scores.append( f[7]             )
        product_scores.append( f[8:12].mean(0)  )

        for s in product_scores:
            print('%05d,  %0.5f'%(np.argmax(s), np.max(s)))


def run_combine_memmap():

    #save_dir = '/root/share/project/kaggle/cdiscount/data/feature/combined'
    save_dir = '/media/ssd/data/kaggle/cdiscount/feature/combined'


    params = {
        'test_id_1768182'    :  3095080,
        'train_id_v0_7019896': 12283645,
        'valid_id_v0_50000'  :    87648,
        'train_id_v1_7019896': 12284153,
        'valid_id_v1_50000'  :    87140,
    }
    feature_dir = '/root/share/project/kaggle/cdiscount/data/feature'
    split = 'test_id_1768182' #'valid_id_v0_50000' #
    num_images = params[split]

    #memmaps = ['se-inception-v3-180-02b', 'se-resnet50-180-00a', 'se-resnext101-180-03b', 'se-resnext101-180-03b-flip']

    dims = np.array([2048,2048,2688,2688], np.int32)
    memmaps = ['se-resnext101-51c', 'se-resnext101-51c-flip', 'dpn92-180-05a', 'dpn92-180-05a-flip']
    num_memmaps = len(memmaps)

    combined_dim = dims.sum()
    combined_file = save_dir + '/%s/features_%dx%d.uint8.memmap' % (split, num_images, combined_dim)
    if 1:
        os.makedirs(save_dir + '/%s' % split, exist_ok=True)
        write_list_to_file(memmaps, save_dir + '/%s/combined_header.txt' % split)

        combined  = np.memmap(combined_file, dtype='uint8', mode='w+', shape=(num_images, combined_dim))
        start = timer()

        d = 0
        for i, m in enumerate(memmaps):
            dim = dims[i]
            memmap_file = feature_dir + '/%s/%s/features_%dx%d.uint8.memmap' % (m, split, num_images, dim)

            f = np.memmap(memmap_file, dtype='uint8', mode='r', shape=(num_images, dim))
            combined[:, d:d+dim] = f
            d = d+dim

            print ('%d  %-36s'%(i, m), f[0, 0:10], '%0.1f  hr'%((timer() - start) / 60/60))
        pass

        print('ok')
        print('')

    if 1:
        #check file
        print('--combine--')
        combined = np.memmap(combined_file, dtype='uint8', mode='r', shape=(num_images, combined_dim))

        d=0
        for i, m in enumerate(memmaps):
            print('%d  %-36s'%(i, m), combined[0, d:d+10])
            d=d+dims[i]



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_feature_extract()
    #run_combine_memmap()


    print('\nsucess!')

