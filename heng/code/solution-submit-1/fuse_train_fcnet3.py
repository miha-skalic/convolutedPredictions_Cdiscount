import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *
from net.rate import *
from net.loss import *
from utility.file import *

from dataset.cdiscount_feature_set_dataset import *
from dataset.sampler import *

## FcNet3 : gated with dropout
# -------------------------------------------------------------------------------------
#from net.model.cdiscount.fcnet1  import FcNet1 as Net; MEMMAP='combined'; DIM=2048*2+2688*2
from net.model.cdiscount.fcnet3  import FcNet3 as Net; MEMMAP='combined'; DIM=2048*2+2688*2



def train_augment(features,label,index):
    dim   = len(features[0])
    count = len(features)
    for n in range(count,CDISCOUNT_MAX_COUNT):
        features.append(np.zeros(dim, np.float32))
    features = np.array(features)

    tensor = torch.from_numpy(features)
    return tensor,label,index


valid_augment = train_augment


#--------------------------------------------------------------
def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (features, labels, indices) in enumerate(test_loader, 0):
        features = Variable(features,volatile=True).cuda()
        labels   = Variable(labels).cuda()

        logits = data_parallel(net, features)
        probs  = F.softmax(logits, dim=1)
        loss   = F.cross_entropy(logits, labels)
        acc    = top_accuracy(probs, labels, top_k=(1,))#1,5

        batch_size = len(indices)
        test_acc  += batch_size*acc[0][0]
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc



#----------------------#----------------------------------------
def run_train():

    out_dir  = '/root/share/project/kaggle/cdiscount/results/fcnet1-dpn-seresnext-00f' # s_xx1'
    initial_checkpoint = \
        '/root/share/project/kaggle/cdiscount/results/fcnet1-dpn-seresnext-00e/checkpoint/00135000_model.pth'
        #None

    pretrained_file = None
    skip = []

    ## setup  ---------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** cap features to [0,2] **\n')
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tDIM          = %s\n' % DIM)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ------------------------------ -
    log.write('** net setting **\n')
    net = Net(in_shape = DIM, num_classes=CDISCOUNT_NUM_CLASSES)
    net.cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    elif pretrained_file is not None:  #pretrain
        log.write('\tpretrained_file    = %s\n' % pretrained_file)
        #net.load_pretrain_file(pretrained_file,skip)

    #net.load_state_dict( torch.load('/root/share/project/kaggle/cdiscount/results/__submission__/stable-00/excited-resnet50-180-00a/checkpoint/00061000_model.pth'))




    log.write('%s\n\n'%(type(net)))
    # log.write('\n%s\n'%(str(net)), is_terminal=0)
    # log.write(inspect.getsource(net.__init__)+'\n', is_terminal=0)
    # log.write(inspect.getsource(net.forw4)+'\n', is_terminal=0)
    log.write('\n')


    ## optimiser ----------------------------------
    iter_accum  = 1
    batch_size  = 4096 #4096  ##NUM_CUDA_DEVICES*512 #256//iter_accum #512 #2*288//iter_accum

    num_iters   = 1000  *1000
    iter_smooth = 20
    iter_log    = 1000
    iter_valid  = 500
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,500))#1*1000


    #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    LR = None #StepLR([ (0, 0.01),])


    ## optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.001/iter_accum, momentum=0.9, weight_decay=0.0001)


    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = CDiscountFeatureSetDataset(
                                    #'train_id_v0_100000', 'train', NAME, DIM, mode='train',
                                    #'debug_train_id_v0_5000', 'train', NAME, DIM, mode='train',
                                    'train_id_v0_7019896', 'train', MEMMAP, DIM, mode='train',
                                     transform = train_augment)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True)

    valid_dataset = CDiscountFeatureSetDataset(
                                    #'debug_train_id_v0_5000', 'train', NAME, DIM, mode='train',
                                    'valid_id_v0_50000', 'train', MEMMAP, DIM, mode='train',
                                    transform = valid_augment)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)

    log.write('\ttrain_dataset.split = %s\n'%(train_dataset.split))
    log.write('\tvalid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n'%(len(valid_loader)))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\titer_accum  = %d\n'%(iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n'%(batch_size*iter_accum))
    log.write('\n')

    #log.write(inspect.getsource(train_augment)+'\n')
    #log.write(inspect.getsource(valid_augment)+'\n')
    log.write('\n')






    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' momentum=%f\n'% optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n'%str(LR) )

    log.write(' products_per_epoch = %d\n\n'%len(train_dataset))
    log.write('   rate   iter_k   epoch  num_m| valid_loss/acc | train_loss/acc | batch_loss/acc |  time   \n')
    log.write('--------------------------------------------------------------------------------------------\n')


    train_loss  = 0.0
    train_acc   = 0.0
    valid_loss  = 0.0
    valid_acc   = 0.0
    batch_loss  = 0.0
    batch_acc   = 0.0
    rate = 0

    start = timer()
    j = 0
    i = 0


    while  i<num_iters:  # loop over the dataset multiple times
        sum_train_loss = 0.0
        sum_train_acc  = 0.0
        sum = 0

        net.train()
        optimizer.zero_grad()
        for features, labels, indices in train_loader:
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)*CDISCOUNT_MAX_COUNT

            if i % iter_valid==0:
                net.eval()
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.train()

                print('\r',end='',flush=True)
                log.write('%0.4f  %5.1f k  %4.2f  %4.1f | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %s \n' % \
                        (rate, i/1000, epoch, num_products/1000000, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc, \
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            #if 1:
            if i in iter_save:
                #https://discuss.pytorch.org/t/dataparallel-optim-and-saving-correctness/4054
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(i))

                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))



            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)[0]*iter_accum


            # one iteration update  -------------
            features = Variable(features).cuda()
            labels   = Variable(labels).cuda()
            logits   = data_parallel(net, features)
            probs    = F.softmax(logits,dim=1)

            loss = F.cross_entropy(logits, labels)
            acc  = top_accuracy(probs, labels, top_k=(1,))

            # single update
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # accumulated update
            loss.backward()
            if j%iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
            batch_acc  = acc[0][0]
            batch_loss = loss.data[0]
            sum_train_loss += batch_loss
            sum_train_acc  += batch_acc
            sum += 1
            if i%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                train_acc  = sum_train_acc /sum
                sum_train_loss = 0.
                sum_train_acc  = 0.
                sum = 0

            print('\r%0.4f  %5.1f k  %4.2f  %4.1f | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %s  %d,%d, %s' % \
                    (rate, i/1000, epoch, num_products/1000000, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc,
                     time_to_str((timer() - start)/60) ,i,j, str(features.size())), end='',flush=True)
            j=j+1
        pass  #-- end of one data loader -
    pass #-- end of all iterations --

    ## check : load model and re-test
    if 1:
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()


    print('\nsucess!')