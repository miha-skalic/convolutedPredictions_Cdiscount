import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *

from train_dpnet92   import *


test_augment = valid_augment
# def test_augment(image,label,index):
#     tensor = image_to_tensor_transform(image)
#     return tensor,label,index


#--------------------------------------------------------------
# predict as uint8
def scores_to_csv(scores, dataset, save_dir):

    #group by products
    df = dataset.df[['product_id','count']]
    df = df.groupby(['product_id']).agg({'count': 'mean'}).reset_index()
    df['cumcount'] = df['count'].cumsum()

    ids      = df['product_id'].values
    count    = df['count'].values
    cumcount = df['cumcount'].values
    num_products = len(ids)
    num_images   = len(scores)

    assert(df['cumcount'].iloc[[-1]].values[0] == num_images)
    print('')
    print('making submission csv')
    print('\tnum_products=%d'%num_products)
    print('\tnum_images=%d'%num_images)
    print('')

    start  = timer()
    labels = []
    probs  = []
    for n in range(num_products):
        if n%10000==0:
            print('\r\t%10d/%d (%0.0f %%)  %0.2f min'%(n, num_products, 100*n/num_products,
                             (timer() - start) / 60), end='',flush=True)
        num = count[n]
        end = cumcount[n]
        if num==1:
            s = scores[end-1]
        else:
            s = scores[end-num:end].mean(axis=0)
            #s = scores[end-num:end].max(axis=0)

        l = s.argmax()
        labels.append(l)
        probs.append(s[l]/255)
    pass
    print('\n')

    # save results ---
    labels = np.array(labels)
    probs  = np.array(probs)
    df = pd.DataFrame({ '_id' : ids, 'category_id' : labels})
    df['category_id'] = df['category_id'].map(dataset.label_to_category_id)

    return df, labels, probs







def run_submit():

    out_dir  = '/root/share/project/kaggle/cdiscount/results/dpn92-180-05'
    checkpoint = out_dir +'/checkpoint/00016500_model.pth'  #final

    ## ------------------------------------
    os.makedirs(out_dir +'/submit', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tcheckpoint   = %s\n'%checkpoint)
    log.write('\n')


    ## net ---------------------------------
    log.write('** net setting **\n')
    log.write('\n')

    net = Net(in_shape = (3,CDISCOUNT_HEIGHT,CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES, type='logits')
    net.load_state_dict(torch.load(checkpoint))
    #net.merge_bn()
    net.cuda()
    net.eval()

    ## dataset  ----------------------------
    test_dataset = CDiscountImageDataset(
                                #'debug_test_id_100', 'test', mode='test',
                                'test_id_1768182', 'test', mode='test',
                                transform = test_augment )
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset), #FixedSampler(test_dataset,[0,1,2,]),  #
                        batch_size  = 256*NUM_CUDA_DEVICES,  #784, 64, #
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)
    num_images = len(test_loader.dataset)


    ## prediction  ----------------------------
    memmap_file = out_dir +'/submit/probs_%dx%d.uint8.memmap'%(num_images,CDISCOUNT_NUM_CLASSES)
    scores = np.memmap(memmap_file, dtype='uint8', mode='w+', shape=(num_images,CDISCOUNT_NUM_CLASSES))


    n = 0
    start = timer()
    for i, (tensors, labels, indices) in enumerate(test_loader, 0):
        print('\r%5d:  %10d/%d (%0.0f %%)  %0.2f min'%(i, n, num_images, 100*n/num_images,
                         (timer() - start) / 60), end='',flush=True)

        # forward
        tensors  = Variable(tensors,volatile=True).cuda(async=True)
        logits = data_parallel(net, tensors)
        probs  = F.softmax(logits, dim=1)

        batch_size = len(indices)
        probs = probs.data.cpu().numpy()
        scores[n:n+batch_size] = probs*255
        n += batch_size


    assert(n == len(test_loader.sampler) and n == num_images)
    print('\r%5d:  %10d/%d (%0.0f %%)  %0.2f min'%(i, n, num_images, 100*n/num_images,
                         (timer() - start) / 60), end='\n',flush=True)

    #save
    df, labels, probs = scores_to_csv(scores, test_dataset, out_dir +'/submit')
    np.savetxt(out_dir + '/submit/labels.txt',labels, fmt='%d')
    np.savetxt(out_dir + '/submit/probs.txt', probs,  fmt='%0.5f')
    df.to_csv(out_dir + '/submit/submission_csv.gz', index=False, compression='gzip')
    log.write('\tscores_to_csv : %0.2f min\n\n'%((timer() - start) / 60))




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_submit()

    print('\nsucess!')