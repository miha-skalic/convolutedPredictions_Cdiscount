from common import *

## for debug
def dummy_transform(img,text='dummy_transform'):
    print ('\t\t%s',text)
    return img

## custom data transform  -----------------------------------
## https://github.com/pytorch/vision/blob/master/test/preprocess-bench.py
## http://pytorch-zh.readthedocs.io/en/latest/torchvision/models.html
##     All pre-trained models expect input images normalized in the same way,
##     i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
##     where H and W are expected to be atleast 224. The images have to be
##     loaded in to a range of [0, 1] and then normalized using
##     mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

def pytorch_image_to_tensor_transform(image):

    mean = [0.485, 0.456, 0.406 ]
    std  = [0.229, 0.224, 0.225 ]

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image).float().div(255)

    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]

    return tensor

def pytorch_tensor_to_image_transform(tensor):
    mean = [0.485, 0.456, 0.406 ]
    std  = [0.229, 0.224, 0.225 ]

    tensor[0] = tensor[0]*std[0] + mean[0]
    tensor[1] = tensor[1]*std[1] + mean[1]
    tensor[2] = tensor[2]*std[2] + mean[2]


    image = tensor.numpy()*255
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    return image


#--------------------------------------------
##https://github.com/pytorch/vision/pull/27/commits/659c854c6971ecc5b94dca3f4459ef2b7e42fb70
## color augmentation

#brightness, contrast, saturation-------------
#from mxnet code, see: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py

# def to_grayscle(img):
#     blue  = img[:,:,0]
#     green = img[:,:,1]
#     red   = img[:,:,2]
#     grey = 0.299*red + 0.587*green + 0.114*blue
#     return grey

def random_gray(image, u=0.5):
    if random.random() < u:
        coef  = np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
        gray  = np.sum(image * coef,axis=2)
        image = np.dstack((gray,gray,gray))
    return image


def random_brightness(image, limit=[-0.3,0.3], u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        image = alpha*image
        image = np.clip(image, 0, 255)
    return image


def random_contrast(image, limit=[-0.3,0.3], u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = alpha*image  + gray
        image = np.clip(image,0,255)
    return image


def random_saturation(image, limit=[-0.3,0.3], u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        coef  = np.array([[[0.114, 0.587,  0.299]]])
        gray  = image * coef
        gray  = np.sum(gray,axis=2, keepdims=True)
        image = alpha*image  + (1.0 - alpha)*gray
        image = np.clip(image,0,255)
    return image

# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# https://github.com/fchollet/keras/pull/4806/files
# https://zhuanlan.zhihu.com/p/24425116
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
def random_hue(image, hue_limit=(-0.1,0.1), u=0.5):
    if random.random() < u:
        h = int(random.uniform(hue_limit[0], hue_limit[1])*180)
        #print(h)

        image = (image*255).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)/255
    return image


#--------------------------------------------
def fix_crop(image, roi=[0,0,256,256]):
    x0,y0,x1,y1 = roi
    image = image[y0:y1,x0:x1,:]
    return image

def fix_resize(image, w, h):
    image = cv2.resize(image,(w,h))
    return image

def random_horizontal_flip(image, u=0.5):
    if random.random() < u:
        image = cv2.flip(image,1)  #np.fliplr(img) ##left-right
    return image


def random_rotate90(image, u=0.5):
    if random.random() < u:

        angle=random.randint(1,3)*90
        if angle == 90:
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            img = cv2.flip(image,1)
            #return img.transpose((1,0, 2))[:,::-1,:]
        elif angle == 180:
            img = cv2.flip(image,-1)
            #return img[::-1,::-1,:]
        elif angle == 270:
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            img = cv2.flip(image,0)
            #return  img.transpose((1,0, 2))[::-1,:,:]

    return image


def random_shift_scale_rotate(image, shift_limit=[-0.0625,0.0625], scale_limit=[1/1.2,1.2],
                               rotate_limit=[-15,15], aspect_limit = [1,1],  size=[-1,-1], borderMode=cv2.BORDER_REFLECT_101 , u=0.5):
    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    if random.random() < u:
        height,width,channel = image.shape
        if size[0]==-1: size[0]=width
        if size[1]==-1: size[1]=height

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(scale_limit[0],scale_limit[1])
        aspect = random.uniform(aspect_limit[0],aspect_limit[1])
        sx    = scale*aspect/(aspect**0.5)
        sy    = scale       /(aspect**0.5)
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)

        image = cv2.warpPerspective(image, mat, (size[0],size[1]),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return image





def random_crop_scale (image, scale_limit=[1/1.2,1.2], size=[-1,-1], u=0.5):
    if random.random() < u:
        image=image.copy()

        height,width,channel = image.shape
        sw,sh = size
        if sw==-1: sw = width
        if sh==-1: sh = height
        box0 = np.array([ [0,0], [sw,0],  [sw,sh], [0,sh], ])

        scale = random.uniform(scale_limit[0],scale_limit[1])
        w = int(scale * sw)
        h = int(scale * sh)

        if w>width and h>height:
            x0 = random.randint(width-w, 0)
            y0 = random.randint(height-h,0)
            x1 = x0+w
            y1 = y0+h
        elif w<width and h<height:
            x0 = random.randint(0, width-w)
            y0 = random.randint(0, height-h)
            x1 = x0+w
            y1 = y0+h
        else:
            raise NotImplementedError
            #pass

        box1 = np.array([ [x0,y0], [x1,y0],  [x1,y1], [x0,y1], ])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box1,box0)

        #<debug>
        # cv2.rectangle(image,(0,0),(width-1,height-1),(0,0,255),1)
        # cv2.rectangle(image,(x0,y0),(x1,y1),(0,255,0),3)
        # print(x0,y0,x1,y1)
        image = cv2.warpPerspective(image, mat, (sw,sh),flags=cv2.INTER_LINEAR, #cv2.BORDER_REFLECT_101
                                    borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return image




# multi crop ----------------------------------
def fix_multi_crop(image, roi_size=(160,160)):

    height, width = image.shape[0:2]
    # assert(height==180)
    # assert(width ==180)

    h,w = roi_size
    dy = height-h
    dx = width -w

    images = []
    rois   = [
            (dx//2, dy//2, width-dx//2,height-dy//2),
            ( 0,     0,    w,       h),
            (dx,     0, width,      h),
            ( 0,    dy,     w, height),
            (dx,    dy, width, height),
        ]

    # for is_flip in [False, True]:
    # #for is_flip in [False]:
    #     if is_flip==True:
    #         image = cv2.flip(image,1)
        #----------------------

    if 1:
        image = cv2.flip(image,1)

        for roi in rois:
            x0,y0,x1,y1 = roi
            i = np.ascontiguousarray(image[y0:y1, x0:x1, :])
            images.append(i)

        # i = image.copy()
        # images.append(i)

        i = cv2.resize(image,roi_size)
        images.append(i)

        # i =  cv2.flip(i,1)
        # images.append(i)

    return images

## -------------------------------------------------------------------------------

#
# from dataset.cdiscount_dataset import *
# def run_check_multi_crop():
#
#     def test_augment(image):
#         images = fix_multi_crop(image, roi_size=(160,160))
#         tensors=[]
#         for image in images:
#             tensor = pytorch_image_to_tensor_transform(image)
#             tensors.append(tensor)
#
#         return tensors
#
#     dataset = CDiscountDataset( #'train_id_v0_5655916', 'train', mode='test',
#                                 'debug_train_id_v0_5000', 'train', mode='test',
#                                    transform=[
#                                        #lambda x: fix_multi_crop(x),
#                                        lambda x: test_augment(x),
#                                     ],
#                                 )
#     sampler = SequentialSampler(dataset)
#     loader  = DataLoader(
#                 dataset,
#                 sampler     = SequentialSampler(dataset),
#                 batch_size  = 16,  #880, #784,
#                 drop_last   = False,
#                 num_workers = 4,
#                 pin_memory  = True)
#
#
#     for i, (images, indices) in enumerate(loader, 0):
#
#         batch_size   = len(indices)
#         num_augments = len(images)
#         print('batch_size = %d'%batch_size)
#         print('num_augments = %d'%num_augments)
#         for a in range(num_augments):
#             tensor = images[a][0]
#             print('%d: %s'%(a,str(tensor.size())))
#             image= pytorch_tensor_to_image_transform(tensor)
#             im_show('image%d'%a,image)
#         cv2.waitKey(0)
#         xx=0


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_multi_crop()


    print('\nsucess!')