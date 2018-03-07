import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *
from net.rate import *
from net.loss import *
from utility.file import *

from dataset.cdiscount_feature_set_dataset import *
from dataset.sampler import *
 
# -------------------------------------------------------------------------------------
#from net.model.cdiscount.fcnet  import FcNet as Net; NAME='SEResNet50'; DIM=2048
#from net.model.cdiscount.fcnet3  import FcNet3 as Net; MEMMAP='combined'; DIM=2048*4
#from net.model.cdiscount.fcnet3  import FcNet3 as Net; MEMMAP='combined'; DIM=2048*2+2688*2

#from net.model.cdiscount.fcnet1  import FcNet1 as Net; MEMMAP='combined'; DIM=2048*2+2688*2
from net.model.cdiscount.fcnet0  import FcNet0 as Net; MEMMAP='combined'; DIM=2048*2+2688*2

def train_augment(features,label,index):
    dim   = len(features[0])
    count = len(features)
    for n in range(count,CDISCOUNT_MAX_COUNT):
        features.append(np.zeros(dim, np.float32))
    features = np.array(features)  ##.max(0)##

    tensor = torch.from_numpy(features)
    return tensor,label,index

valid_augment = train_augment
test_augment  = train_augment
# ----------------------------------------------------------------
 
 

def run_submit():

    #out_dir  = '/root/share/project/kaggle/cdiscount/results/gated-combined4-05a' # s_xx1'
    #checkpoint = '/root/share/project/kaggle/cdiscount/results/gated-combined4-05a/checkpoint/00135000_model.pth'

    #out_dir  = '/root/share/project/kaggle/cdiscount/results/fcnet3-dpn-seresnext-00c' # s_xx1'
    #checkpoint = '/root/share/project/kaggle/cdiscount/results/fcnet3-dpn-seresnext-00c/checkpoint/00131000_model.pth'

    # out_dir  = '/root/share/project/kaggle/cdiscount/results/fcnet1-dpn-seresnext-00f' # s_xx1'
    # checkpoint = '/root/share/project/kaggle/cdiscount/results/fcnet1-dpn-seresnext-00f/checkpoint/00150000_model.pth'


    out_dir  = '/root/share/project/kaggle/cdiscount/results/fcnet0-dpn-seresnext-soft-02a' # s_xx1'
    checkpoint = '/root/share/project/kaggle/cdiscount/results/fcnet0-dpn-seresnext-soft-02a/checkpoint/00095000_model.pth'


    csv_file    = out_dir +'/submit/submission_csv.gz'
    memmap_file = out_dir +'/submit/probs.uint8.memmap'

    ## ------------------------------------
    os.makedirs(out_dir +'/submit', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%ID)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tDIM          = %s\n' % DIM) 
    log.write('\tcheckpoint   = %s\n'%checkpoint)
    log.write('\n')


    ## net ---------------------------------
    log.write('** net setting **\n') 

    net = Net(in_shape =DIM, num_classes=CDISCOUNT_NUM_CLASSES) 
    net.load_state_dict(torch.load(checkpoint))
    net.cuda()
    net.eval()
 

    log.write('\tcheckpoint = %s\n' % checkpoint)
    log.write('%s\n\n'%(type(net)))

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
 
    test_dataset = CDiscountFeatureSetDataset(
                                    'test_id_1768182', 'test', MEMMAP, DIM, mode='test',
                                     transform = test_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),  
                        batch_size  = 256*NUM_CUDA_DEVICES,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)

    test_num  = len(test_loader.dataset)
    start = timer()


    norm_probs = np.memmap(memmap_file, dtype='uint8', mode='w+', shape=(test_num, CDISCOUNT_NUM_CLASSES))
    categories = np.zeros(test_num, np.int32) 
    start = timer()

    n = 0
    for features, labels, indices in test_loader:
        print('\rpredicting: %10d/%d (%0.0f %%)  %0.2f min'%(n, test_num, 100*n/test_num,
                         (timer() - start) / 60), end='',flush=True)
        time.sleep(0.01)

        # forward
        features = Variable(features,volatile=True).cuda(async=True)
        logits   = data_parallel(net, features)
        probs    = F.softmax(logits,dim=1)
        labels   = probs.topk(1)[1]

        labels = labels.data.cpu().numpy().reshape(-1)
        probs  = probs.data.cpu().numpy()*255
        probs  = probs.astype(np.uint8)

        batch_size = len(indices)
        categories[n:n+batch_size]=labels
        norm_probs[n:n+batch_size]=probs
        n += batch_size

    print('\n')
    assert(n == len(test_loader.sampler) and n == test_num)

 
    ## submission  ----------------------------
    df = pd.DataFrame({ '_id' : test_dataset.product_ids , 'category_id' : categories})
    df['category_id'] = df['category_id'].map(test_dataset.label_to_category_id)
    df.to_csv(csv_file, index=False, compression='gzip')
 



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_submit()


    print('\nsucess!')
