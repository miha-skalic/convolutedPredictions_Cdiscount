from common import *


def run_check_cvs():
    df1 = pd.read_csv('/root/share/project/kaggle/cdiscount/results/__mihas__/blend/00/heng_en52rt1.csv.gz',   compression='gzip')
    df2 = pd.read_csv('/root/share/project/kaggle/cdiscount/results/__mihas__/blend/00/heng_en52rt.csv.gz', compression='gzip')
    df1 = df1.sort_values('_id')
    df2 = df2.sort_values('_id')
    print('check same file :', df1.equals(df2))
    print('check same _id  :', (df1['_id'].values == df2['_id'].values ).sum()==1768182)
    print('')
    same = df1['category_id'].values  == df2['category_id'].values
    same_percent = same.sum()/len(same)
    print('all  = %d'%len(same))
    print('same (%%) = %7d  (%0.5f)'%(same.sum(),same_percent))
    print('diff (%%) = %7d  (%0.5f)'%(len(same)-same.sum(),1-same_percent))
    pass



def run_blend_52rt():

    num_ids = 1768182
    CDISCOUNT_NUM_CLASSES = 5270
    label_to_category_id = pickle.load(open('/media/root/5453d6d1-e517-4659-a3a8-d0a878ba4b60/data/kaggle/cdiscount/label_to_category_id.pkl', 'rb'))
    ids = np.loadtxt('/media/root/5453d6d1-e517-4659-a3a8-d0a878ba4b60/data/kaggle/cdiscount/split/test_id_1768182', np.int64)

    #results
    weight_sum = 0
    scores = np.zeros((num_ids,CDISCOUNT_NUM_CLASSES), np.float32)
    csv_file = '/root/share/project/kaggle/cdiscount/results/__mihas__/blend/00/heng_en52rt.csv.gz'



    ## mihas's memmap ##--------------------------------------------
    h_order = dict(zip(list(ids), list(range(num_ids))))
    m_order = pickle.load(open(
        '/root/share/project/kaggle/cdiscount/results/__mihas__/idorder_layer2_xception_inceptionresnet_inceptionv3_8k12k_cp300_wflips.pkl',
        'rb'))
    m_order = dict(zip(m_order, list(range(num_ids))))



    m_memmap1 = '/root/share/project/kaggle/cdiscount/results/__mihas__/mihas_3model_c100_DO03_wflip.memmap'
    m_memmap2 = '/root/share/project/kaggle/cdiscount/results/__mihas__/mihas_3model_c100_width10k_DO03_wflip.memmap'
    m_blends=[
        [0.1323529411764706, np.memmap( m_memmap1,      dtype='uint8', mode='r', shape=(num_ids, CDISCOUNT_NUM_CLASSES) ) ],
        [0.1323529411764706, np.memmap( m_memmap2,      dtype='uint8', mode='r', shape=(num_ids, CDISCOUNT_NUM_CLASSES) ) ],
    ]

    for n, (weight, memmap) in enumerate(m_blends):
        scores += weight * (memmap.astype(np.float32)**0.5)
        weight_sum += weight
        print('m_blends : n = %d'%n)

    #reorder
    index = []
    for n in range(num_ids):
        index.append(m_order[ids[n]])
    scores = scores[index]


    ## heng's memmap ##--------------------------------------------
    h_memmap0 = '/root/share/project/kaggle/cdiscount/[solution]/combine.0/gated-combined4-00c/submit/probs.uint8.memmap'
    h_memmap1 = '/root/share/project/kaggle/cdiscount/[solution]/combine.1/gated-combined4-drop0.3-01a/submit/probs.uint8.memmap'
    h_memmap2 = '/root/share/project/kaggle/cdiscount/[solution]/combine.2/gated-combined4-05a/submit/probs.uint8.memmap'
    h_memmap5 = '/root/share/project/kaggle/cdiscount/[solution]/combine.5/fcnet3-dpn-seresnext-00b/submit/probs.uint8.memmap'
    h_memmap5_2   = '/root/share/project/kaggle/cdiscount/[solution]/combine.5-2/fcnet3-dpn-seresnext-00c/submit/probs.uint8.memmap'
    h_memmap5_max = '/root/share/project/kaggle/cdiscount/results/fcnet1-dpn-seresnext-00f/submit/probs.uint8.memmap'
    h_memmap5_mix = '/root/share/project/kaggle/cdiscount/results/fcnet0-dpn-seresnext-soft-02a/submit/probs.uint8.memmap'
    h_blends=[
        [0.1, np.memmap( h_memmap0,      dtype='uint8', mode='r', shape=(num_ids, CDISCOUNT_NUM_CLASSES) ) ],
        [0.1, np.memmap( h_memmap1,      dtype='uint8', mode='r', shape=(num_ids, CDISCOUNT_NUM_CLASSES) ) ],
        [0.1, np.memmap( h_memmap2,      dtype='uint8', mode='r', shape=(num_ids, CDISCOUNT_NUM_CLASSES) ) ],
        #[0.11764705882352942, np.memmap( h_memmap5,      dtype='uint8', mode='r', shape=(num_ids, CDISCOUNT_NUM_CLASSES) ) ],
        [0.1, np.memmap( h_memmap5_2,    dtype='uint8', mode='r', shape=(num_ids, CDISCOUNT_NUM_CLASSES) ) ],
        [0.1, np.memmap( h_memmap5_max,  dtype='uint8', mode='r', shape=(num_ids, CDISCOUNT_NUM_CLASSES) ) ],
        [0.1, np.memmap( h_memmap5_mix,  dtype='uint8', mode='r', shape=(num_ids, CDISCOUNT_NUM_CLASSES) ) ],
    ]
    for n, (weight, memmap) in enumerate(h_blends):
        scores += weight *(memmap.astype(np.float32)**0.5)
        weight_sum += weight
        print('h_blends : n = %d'%n)


    ## outrunner's memmap ##--------------------------------------------
    o_memmap0 = '/root/share/project/kaggle/cdiscount/results/__mihas__/ensemble_all.mm'  # i.e. 'outrunner_ensemble_all_upload4.mm'
    o_blends = [
        [0.2647058823529412, np.memmap(o_memmap0, dtype='uint8', mode='r', shape=(num_ids, CDISCOUNT_NUM_CLASSES))],
    ]
    for n, (weight, memmap) in enumerate(o_blends):
        scores += weight *(memmap.astype(np.float32)**0.5)
        weight_sum += weight
        print('h_blends : n = %d'%n)


    ## submission  ----------------------------
    scores = scores / weight_sum / 255
    categories = scores.argmax(1)

    product_ids = ids
    df = pd.DataFrame({ '_id' : product_ids , 'category_id' : categories})
    df['category_id'] = df['category_id'].map(label_to_category_id)
    df.to_csv(csv_file, index=False, compression='gzip')






# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_blend_52rt2()

    print('\nsucess!')