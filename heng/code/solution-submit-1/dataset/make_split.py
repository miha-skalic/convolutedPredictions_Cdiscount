from common import *
from dataset.tool import *
from utility.file import *

import bson
CDISCOUNT_DIR = '/media/ssd/data/kaggle/cdiscount'

## https://qiita.com/wasnot/items/be649f289073fb96513b
## https://www.kaggle.com/inversion/processing-bson-files
## https://www.kaggle.com/carlossouza/extract-image-files-from-bson-and-save-to-disk/code
## test Processing BSON Files


def run_make_train_summary():

    bson_file = '/media/ssd/data/kaggle/cdiscount/__download__/train.bson'
    num_products= 7069896 # 7069896 for train and 1768182 for test
    out_dir = CDISCOUNT_DIR

    id = []
    num_imgs = []
    category_id = []

    with open(bson_file, 'rb') as fbson:
        data = bson.decode_file_iter(fbson)
        #num_products = len(list(data))
        #print ('num_products=%d'%num_products)
        #exit(0)

        for n, d in enumerate(data):
            print('\r%08d/%08d'%(n,num_products), flush=True,end='')

            category_id.append(d['category_id'])
            id.append(d['_id'])
            num_imgs.append(len(d['imgs']))
        print('')

    #by product id
    df = pd.DataFrame({ '_id' : id, 'num_imgs' : num_imgs, 'category_id' : category_id})
    df.to_csv('/media/ssd/data/kaggle/cdiscount/__temp__/train_by_product_id.csv', index=False)
    t = df['num_imgs'].sum()  #check :12371293
    print(t)

    #split by id --------------------------------------
    id_random = list(id)
    random.shuffle(id_random)

    #make train, valid
    num_valid = 50000
    num_train = num_products - num_valid

    #by id
    file1 = CDISCOUNT_DIR +'/split/'+ 'train_id_v0_%d'%(num_train)
    file2 = CDISCOUNT_DIR +'/split/'+ 'valid_id_v0_%d'%(num_valid)
    id1 = id_random[0:num_train]
    id2 = id_random[num_train: ]
    write_list_to_file(id1, file1)
    write_list_to_file(id2, file2)


    #summary ------------------------------------
    # g=(df.groupby('category_id')
    #    .agg({'_id':'count', 'num_imgs': 'sum'})
    #    .reset_index()
    # )
    # g.to_csv('/media/ssd/data/kaggle/cdiscount/__temp__/train_g.csv', index=False)



def run_make_test_summary():

    bson_file = '/media/ssd/data/kaggle/cdiscount/__download__/test.bson'
    num_products= 1768182 # 7069896 for train and 1768182 for test
    out_dir = CDISCOUNT_DIR

    id = []
    num_imgs = []
    with open(bson_file, 'rb') as fbson:
        data = bson.decode_file_iter(fbson)

        for n, d in enumerate(data):
            print('\r%08d/%08d'%(n,num_products), flush=True,end='')

            id.append(d['_id'])
            num_imgs.append(len(d['imgs']))
        print('')

    #by product id
    df = pd.DataFrame({ '_id' : id, 'num_imgs' : num_imgs })
    df.to_csv('/media/ssd/data/kaggle/cdiscount/__temp__/test_by_product_id.csv', index=False)
    t = df['num_imgs'].sum()  #check :12371293
    print('total num of images = %d'%t)


    #by id
    num_test = num_products
    file = CDISCOUNT_DIR +'/split/'+ 'test_id_%d'%(num_test)
    write_list_to_file(id, file)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_make_train_summary()
    run_make_test_summary ()

