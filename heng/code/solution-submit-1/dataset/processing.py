from common import *
from utility.file import *


# main #################################################################
# quantisation:
#     https://arxiv.org/pdf/1511.06393.pdf
#     https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/
#     https://openreview.net/pdf?id=rJ8uNptgl
#
#     http://openaccess.thecvf.com/content_cvpr_2017/papers/Park_Weighted-Entropy-Based_Quantization_for_CVPR_2017_paper.pdf
#     https://arxiv.org/pdf/1603.01025.pdf  (log quant)
#     https://arxiv.org/pdf/1702.03044.pdf
#


def run_check_histogram():
    memmap_file = '/media/ssd/data/kaggle/cdiscount/feature/se-resnext101/train_id_v0_7019896/features_12283645x2048.float16.memmap'
    feature = np.memmap(memmap_file, dtype='float16', mode='r', shape=(12283645,2048))
    feature = np.array(feature[:10000]).reshape(-1).astype(np.float64)

    print('max  ', feature.max())
    print('min  ', feature.min())
    print('mean ', feature.mean())
    print('std  ', feature.std())

    q0 = np.log2(feature+1e-9)

    b_min = 0
    b_max = 1.5
    bins = np.arange(0.,1.,1./256)
    hist, bin_edges = np.histogram(feature, bins, normed=True)
    zz=0

    plt.plot(hist)
    plt.show()

def run_check_overlap():
    split1 = '/media/ssd/data/kaggle/cdiscount/split/valid_id_v1_5000'
    split2 = '/media/ssd/data/kaggle/cdiscount/split/train_id_v0_7019896'
    ids1 = read_list_from_file(split1, comment='#', func=int)
    ids2 = read_list_from_file(split2, comment='#', func=int)

    num=0
    N=len(ids2)

    ids1 = np.array(ids1)
    ids2 = np.array(ids2)
    #ids1 = np.array(ids2)
    for n,id in enumerate(ids2):
        num += len(np.where((ids1-id) == 0)[0])
        print('%10d   %10d   (%8d/%8d , %f)'%(id,num,n,N,n/N))


    print('')
    print(num)


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_overlap()

    print('\nsucess!')
