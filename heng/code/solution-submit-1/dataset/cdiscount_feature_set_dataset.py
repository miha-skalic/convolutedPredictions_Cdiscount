##  https://www.kaggle.com/c/cdiscount-image-classification-challenge
##  Cdiscount’s Image Classification Challenge
##
##  https://www.kaggle.com/alekseit/pytorch-bson-dataset
##  https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson



from dataset.transform import *
from dataset.sampler import *
from utility.file import *
from utility.draw import *
import bson
import struct


CDISCOUNT_DIR = '/media/ssd/data/kaggle/cdiscount'
CDISCOUNT_NUM_CLASSES = 5270
CDISCOUNT_HEIGHT=180
CDISCOUNT_WIDTH =180
CDISCOUNT_MAX_COUNT = 4



def encode_features(features):
    v_max=3
    features = np.clip(features,0,v_max)
    features = np.round(features/v_max*255).astype(np.uint8)
    return features


def decode_features(features):
    v_max=3
    features = features.astype(np.float32)/255*v_max
    return features


#
# #data iterator ----------------------------------------------------------------
class CDiscountFeatureSetDataset(Dataset):

    def __init__(self, split, bson, memmap, dim, transform=None, mode='train'):
        super(CDiscountFeatureSetDataset, self).__init__()
        self.split = split
        self.name  = bson
        self.mode  = mode
        self.transform = transform

        #label to name
        category_names_df = pd.read_csv (CDISCOUNT_DIR + '/category_names.csv')
        category_names_df['label'] = category_names_df.index
        label_to_category_id = dict(zip(category_names_df['label'], category_names_df['category_id']))
        category_id_to_label = dict(zip(category_names_df['category_id'], category_names_df['label']))

        self.category_names_df    = category_names_df
        self.label_to_category_id = label_to_category_id
        self.category_id_to_label = category_id_to_label

        #read split
        print('read img list')
        ids = read_list_from_file(CDISCOUNT_DIR + '/split/' + split, comment='#', func=int)
        num_ids = len(ids)


        #read images and labels
        start = timer()
        df = pd.read_csv (CDISCOUNT_DIR + '/%s_by_product_id.csv'%bson)
        #df.columns #(['product_id', 'category_id', 'count', 'offset', 'length'], dtype='object')

        df = df.reset_index()
        df = df[ df['product_id'].isin(ids)]


        if mode=='train':
            df['label'] = df['category_id'].map(category_id_to_label)
            self.labels = list(df['label'])
        elif mode=='test':
            self.labels = None
        else:
            raise NotImplementedError

        assert(num_ids == df['product_id'].nunique())
        num_images = df['count'].sum()
        self.product_ids = list(df['product_id'])
        self.ids = ids
        self.df  = df

        self.cumcounts = [0,]+ list(np.cumsum(np.array(df['count'].values)))
        memmap_file = CDISCOUNT_DIR + '/feature/%s/%s/features_%dx%d.uint8.memmap'%(memmap,split,num_images,dim)
        self.features= np.memmap(memmap_file, dtype='uint8', mode='r', shape=(num_images,dim))

        #save
        print('\tnum_ids    = %d'%(num_ids))
        print('\tnum_images = %d'%(num_images))
        print('\ttime = %0.2f min'%((timer() - start) / 60))
        print('')


    def __getitem__(self, index):
        start = self.cumcounts[index]
        end   = self.cumcounts[index+1]
        features = []
        for i in range(start,end):
            features.append(decode_features(self.features[i]))
            #features.append(self.features[i])

        if self.mode=='train':
            label = self.labels[index]
        elif self.mode=='test':
            label = None

        if self.transform is not None:
            return self.transform(features, label, index)

        return features, label, index


    def __len__(self):
        return len(self.ids)




## check ## ----------------------------------------------------------
## https://github.com/hujie-frank/SENet
##   Random Mirror	True
##   Random Crop	8% ~ 100%
##   Aspect Ratio	3/4 ~ 4/3
##   Random Rotation -10° ~ 10°
##   Pixel Jitter	 -20 ~ 20  (shift)
##


def run_check_dataset():

    dataset = CDiscountFeatureSetDataset(
                                'valid_id_v0_50000', 'train', 'se-resnet50-180-00a', 2048, mode='train',
                                transform = None,)
    sampler = SequentialSampler(dataset)


    print('index, str(label), str(image.shape)')
    print('-----------------------------------')
    #for n in iter(sampler):
    for n in range(10):
        features, label, index = dataset[n]
        print('%09d %s %d x %s   '%(index, str(label), len(features), str(features[0].shape)), end='')
        print(features[0][0:5])





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    #train_bson_to_summary_cvs()
    #test_bson_to_summary_cvs()

    run_check_dataset()
